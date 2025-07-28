"""
Dataset handling for CIFake dataset
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random
import shutil


class CIFakeDataset(Dataset):
    """CIFake Dataset for AI-generated image detection"""
    
    def __init__(self, data_dir, split='train', transform=None, target_transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Define paths for real and fake images
        self.real_dir = os.path.join(data_dir, 'real')
        self.fake_dir = os.path.join(data_dir, 'fake')
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        # Load real images (label 0)
        if os.path.exists(self.real_dir):
            real_images = [os.path.join(self.real_dir, f) for f in os.listdir(self.real_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.images.extend(real_images)
            self.labels.extend([0] * len(real_images))
        
        # Load fake images (label 1)
        if os.path.exists(self.fake_dir):
            fake_images = [os.path.join(self.fake_dir, f) for f in os.listdir(self.fake_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.images.extend(fake_images)
            self.labels.extend([1] * len(fake_images))
        
        print(f"Loaded {len(self.images)} images for {split} split")
        print(f"Real images: {self.labels.count(0)}, Fake images: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            # Albumentations expects a dict with 'image'
            transformed = self.transform(image=np.array(image))
            image = transformed['image']
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def get_transforms(config, split='train'):
    """Get data transforms for different splits"""
    
    if split == 'train':
        # Enhanced augmentation for training - more sophisticated for AI detection
        transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=5),
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
            ], p=0.4),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            ], p=0.4),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.3, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.5),
            ], p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # Minimal augmentation for validation/test
        transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    return transform


def create_dataloaders(config, data_dir):
    """Create train, validation, and test dataloaders"""
    
    # Get all image paths
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    real_images = []
    fake_images = []
    
    if os.path.exists(real_dir):
        real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if os.path.exists(fake_dir):
        fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle and split
    random.seed(config.SEED)
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    # Calculate split indices
    total_real = len(real_images)
    total_fake = len(fake_images)
    
    val_real = int(total_real * config.VAL_SPLIT)
    test_real = int(total_real * config.TEST_SPLIT)
    train_real = total_real - val_real - test_real
    
    val_fake = int(total_fake * config.VAL_SPLIT)
    test_fake = int(total_fake * config.TEST_SPLIT)
    train_fake = total_fake - val_fake - test_fake
    
    # Split data
    train_real_imgs = real_images[:train_real]
    val_real_imgs = real_images[train_real:train_real + val_real]
    test_real_imgs = real_images[train_real + val_real:]
    
    train_fake_imgs = fake_images[:train_fake]
    val_fake_imgs = fake_images[train_fake:train_fake + val_fake]
    test_fake_imgs = fake_images[train_fake + val_fake:]
    
    # Create temporary directories for splits
    os.makedirs(os.path.join(data_dir, 'temp_train', 'real'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'temp_train', 'fake'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'temp_val', 'real'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'temp_val', 'fake'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'temp_test', 'real'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'temp_test', 'fake'), exist_ok=True)

    # Create copies for train split
    for img_path in train_real_imgs:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(data_dir, 'temp_train', 'real', filename))

    for img_path in train_fake_imgs:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(data_dir, 'temp_train', 'fake', filename))

    # Create copies for validation split
    for img_path in val_real_imgs:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(data_dir, 'temp_val', 'real', filename))

    for img_path in val_fake_imgs:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(data_dir, 'temp_val', 'fake', filename))

    # Create copies for test split
    for img_path in test_real_imgs:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(data_dir, 'temp_test', 'real', filename))

    for img_path in test_fake_imgs:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(data_dir, 'temp_test', 'fake', filename))
    
    # Create datasets
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    
    train_dataset = CIFakeDataset(
        os.path.join(data_dir, 'temp_train'), 
        'train', 
        transform=train_transform
    )
    
    val_dataset = CIFakeDataset(
        os.path.join(data_dir, 'temp_val'), 
        'val', 
        transform=val_transform
    )
    
    test_dataset = CIFakeDataset(
        os.path.join(data_dir, 'temp_test'), 
        'test', 
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def cleanup_temp_dirs(data_dir):
    """Clean up temporary directories"""
    temp_dirs = ['temp_train', 'temp_val', 'temp_test']
    for temp_dir in temp_dirs:
        temp_path = os.path.join(data_dir, temp_dir)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path) 