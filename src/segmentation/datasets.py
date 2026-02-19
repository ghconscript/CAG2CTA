import os

import cv2
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset

# ===============================
# 路径统一管理（关键）
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # segmentation/
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))   # src/
DATASET_DIR = os.path.join(SRC_DIR, 'datasets')


class SimpleDatasets(Dataset):
    """
    用于 FR-UNet 的 2D 血管分割 Dataset
    image: [1, H, W], float32, [0,1]
    mask : [1, H, W], float32, {0,1}
    """

    def __init__(self, split='train', image_size=512):
        assert split in ['train', 'val', 'test']
        self.image_size = image_size

        self.image_dir = os.path.join(DATASET_DIR, f'{split}_img')
        self.mask_dir = os.path.join(DATASET_DIR, f'{split}_mask')

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f'图像路径不存在: {self.image_dir}')
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f'标签路径不存在: {self.mask_dir}')

        self.img_list = natsorted(os.listdir(self.image_dir))
        self.mask_list = natsorted(os.listdir(self.mask_dir))

        assert len(self.img_list) == len(self.mask_list), '图像与标签数量不一致'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.img_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f'无法读取图像: {img_path}')
        if mask is None:
            raise ValueError(f'无法读取标签: {mask_path}')

        # resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # 二值化 mask
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

        # [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image).float() / 255.0
        mask = torch.from_numpy(mask).float() / 255.0

        return image, mask
