import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from nets.geom import *
from nets.voxel import *
from nets.deformableattention import DeformableAttention2D
from nets.acmf import AdaptiveCrossModalityFusion

class encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        
    def forward(self, x):
        x = self.backbone(x)
        return x

def splat_sample_to_bev(feats, coords_mem, Z, Y, X):
        B,S,C,D,H,W = feats.shape
        output = torch.zeros((B,C,Z,X), dtype=torch.float, device=feats.device)
        output_ones = torch.zeros((B,1,Z,X), dtype=torch.float, device=feats.device)

        feats = feats.permute(0,1,3,4,5,2)

        N = S * D * H * W 
        for b in range(B):

            x_b = feats[b].reshape(N, C)

            coords_mem_b = coords_mem[b].reshape(N, 3).long()

            valid = (
                (coords_mem_b[:, 0] >= 0)
                & (coords_mem_b[:, 0] < X)
                & (coords_mem_b[:, 1] >= 0)
                & (coords_mem_b[:, 1] < Y)
                & (coords_mem_b[:, 2] >= 0)
                & (coords_mem_b[:, 2] < Z)
            )
            x_b = x_b[valid]
            coords_mem_b = coords_mem_b[valid]

            inds = (
                coords_mem_b[:, 2] * Y * X
                + coords_mem_b[:, 1] * X
                + coords_mem_b[:, 0]
            )
            sorting = inds.argsort()
            x_b, coords_mem_b, inds = x_b[sorting], coords_mem_b[sorting], inds[sorting]

            one_b = torch.ones_like(x_b[:,0:1])
            x_b = x_b / one_b.clamp(min=1.0)

            bev_feature = torch.zeros((Z,Y,X,C), device=x_b.device)
            bev_ones = torch.zeros((Z,Y,X,1), device=x_b.device)
            bev_feature[coords_mem_b[:,2],coords_mem_b[:,1],coords_mem_b[:,0]] = x_b # Z,Y,X,C
            bev_ones[coords_mem_b[:,2],coords_mem_b[:,1],coords_mem_b[:,0]] = one_b # Z,Y,X,C
            bev_feature = bev_feature.sum(dim=1) # Z,X,C
            bev_feature = bev_feature.permute(2,0,1) # C,Z,X

            bev_ones = bev_ones.sum(dim=1) # Z,X,C
            bev_ones = bev_ones.permute(2,0,1) # C,Z,X
            
            output[b] = bev_feature
            output_ones[b] = bev_ones
        output = output / output_ones.clamp(min=1)
        return output

class FLSFusionNet(nn.Module):
    def __init__(self, num_classes, X, Y, Z,):
        super(FLSFusionNet, self).__init__()
        self.X = X
        self.Y = Y
        self.Z = Z
        self.camera_backbone = encoder_res50(256)
        self.sonar_backbone = encoder_res50(256)
        XMIN, XMAX = -50, 50
        ZMIN, ZMAX = -50, 50
        
        YMIN, YMAX = -5, 5
        bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
        scene_centroid_x = 0.0
        scene_centroid_y = 2.0
        scene_centroid_z = 0.0
        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        scene_centroid = torch.from_numpy(scene_centroid_py).float()
        self.vox_util = Vox_util(Z, Y, X, scene_centroid=scene_centroid, bounds=bounds, pad=None, assert_cube=False)
        self.deformable_attention = DeformableAttention2D(512)
        self.adaptive_cross_modality_fusion = AdaptiveCrossModalityFusion(512)
        self.segmentation_head = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, rgb_cam, fls_sonar, cam0_T_cam):
        B, C, H, W = rgb_cam.shape
        assert(C==3)

        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)
        rgb_cam_ = __p(rgb_cam)
        fls_sonar_ = __p(fls_sonar) 
        cam0_T_cam_ = __p(cam0_T_cam)
        cam_T_cam0_ = safe_inverse(cam0_T_cam_)
        _, D, Hf, Wf = cam_T_cam0_.shape
        sy = Hf/float(H)
        sx = Wf/float(W)

        feat_cam_ = self.camera_backbone(rgb_cam_)
        feat_sonar_ = self.sonar_backbone(fls_sonar_)
        
        featpix_T_cams_ = scale_intrinsics(cam_T_cam0_, sx, sy) 
        xyz_mem0s_ = self.vox_util.Ref2Mem(featpix_T_cams_, self.Z, self.Y, self.X, assert_cube=False)
        xyz_mem0s = __u(xyz_mem0s_) 

        feat_bev_cam_ = self.splat_to_bev(feat_cam_, xyz_mem0s, self.Z, self.Y, self.X)

        feat_bev_cam_ = self.deformable_attention(feat_bev_cam_, feat_sonar_)

        feat_fusion_ = self.adaptive_cross_modality_fusion(feat_bev_cam_, feat_sonar_)

        output = self.segmentation_head(feat_fusion_)

        return output

