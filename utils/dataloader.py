import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class FusionDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(FusionDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        camera_jpg      = Image.open(os.path.join(os.path.join(self.dataset_path, "Camera_Images"), name + ".jpg"))
        sonar_jpg       = Image.open(os.path.join(os.path.join(self.dataset_path, "Sonar_Images"), name + ".jpg"))
        png             = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".png"))
        IMU_data        = np.load(os.path.join(os.path.join(self.dataset_path, "IMU_data"), name + ".npy"))

        camera_jpg, sonar_jpg, png  = self.get_random_data(camera_jpg, sonar_jpg, png, self.input_shape, random = self.train)

        camera_jpg  = np.transpose(preprocess_input(np.array(camera_jpg, np.float64)), [2,0,1])
        sonar_jpg   = np.transpose(preprocess_input(np.array(sonar_jpg, np.float64)), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return camera_jpg, sonar_jpg, IMU_data, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, camera_image, sonar_image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        camera_image   = cvtColor(camera_image)
        sonar_image   = cvtColor(sonar_image)
        label   = Image.fromarray(np.array(label))
        iw, ih  = camera_image.size
        h, w    = input_shape

        if not random:
            iw, ih  = camera_image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            camera_image       = camera_image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(camera_image, ((w-nw)//2, (h-nh)//2))

            sonar_image       = sonar_image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(sonar_image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        camera_image = camera_image.resize((nw,nh), Image.BICUBIC)
        sonar_image = sonar_image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        flip = self.rand()<.5
        if flip: 
            camera_image = camera_image.transpose(Image.FLIP_LEFT_RIGHT)
            sonar_image = sonar_image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(camera_image, (dx, dy))
        new_label.paste(label, (dx, dy))
        new_image2 = Image.new('RGB', (w,h), (128,128,128))
        new_image2.paste(sonar_image, (dx, dy))

        camera_image = new_image
        sonar_image = new_image2
        label = new_label

        camera_image_data      = np.array(camera_image, np.uint8)
        sonar_image_data      = np.array(sonar_image, np.uint8)

        blur = self.rand() < 0.25
        if blur: 
            camera_image_data = cv2.GaussianBlur(camera_image_data, (5, 5), 0)
            sonar_image_data = cv2.GaussianBlur(sonar_image_data, (5, 5), 0)

        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            camera_image_data  = cv2.warpAffine(camera_image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            sonar_image_data  = cv2.warpAffine(sonar_image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val   = cv2.split(cv2.cvtColor(camera_image_data, cv2.COLOR_RGB2HSV))
        dtype           = camera_image_data.dtype
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        camera_image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        camera_image_data = cv2.cvtColor(camera_image_data, cv2.COLOR_HSV2RGB)
        
        return camera_image_data, sonar_image_data, label


def deeplab_dataset_collate(batch):
    camera_images      = []
    sonar_images        = []
    IMU_data        = []
    pngs        = []
    seg_labels  = []
    for camera_img, sonar_jpg, IMU_data, png, labels in batch:
        camera_images.append(camera_img)
        sonar_images.append(sonar_jpg)
        IMU_data.append(IMU_data)
        pngs.append(png)
        seg_labels.append(labels)
    camera_images      = torch.from_numpy(np.array(camera_images)).type(torch.FloatTensor)
    sonar_images       = torch.from_numpy(np.array(sonar_images)).type(torch.FloatTensor)
    IMU_data           = torch.from_numpy(np.array(IMU_data)).type(torch.FloatTensor)
    pngs               = torch.from_numpy(np.array(pngs)).long()
    seg_labels         = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return camera_images, sonar_images, IMU_data, pngs, seg_labels
