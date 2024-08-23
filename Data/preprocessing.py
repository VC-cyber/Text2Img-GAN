import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from PIL import Image

# Custom Dataset class to handle COCO captions
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transform = transform
        
        # Define natural landscape categories
        # natural_landscape_categories = [
        #     'mountain', 'hill', 'desert', 'sea', 'lake', 'river', 
        #     'forest', 'field', 'sky', 'tree', 'flower'
        # ]
        # print(self.coco.dataset.keys())

        # # Get category IDs for natural landscapes
        # category_ids = self.coco.getAnnIds(catNms=natural_landscape_categories)
        
        # Get image IDs that contain at least one of the natural landscape categories
        self.ids = list(self.coco.imgToAnns.keys())
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        annotations = self.coco.imgToAnns[img_id]
        caption = random.choice(annotations)['caption']

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Try loading the image; if it fails, return None
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return None

        if self.transform:
            image = self.transform(image)

        return image, caption
