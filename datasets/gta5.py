import os
from PIL import Image
import torch
from torch.utils.data import Dataset

# TODO: implement here your custom dataset class for GTA5
#### VEDERE COME DIVIDERLO IN TRAIN E VAL (POI EVAL CON CITYSCAPES)

class GTA5(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(GTA5, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        # Image and label directories
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')

        # Collect sorted image and label
        self.image_paths = sorted([
            os.path.join(self.image_dir, f)     # Joins the folder path with the filename to get the full path to each image (/content/gta5/GTA5/images/00001.pn)
            for f in os.listdir(self.image_dir) # LISTDIR -> lists all files in the images directory
            if f.endswith('.png')               # Filters only files that are PNG images
        ])
        
        # The same for the labels
        self.label_paths = sorted([
            os.path.join(self.label_dir, f)
            for f in os.listdir(self.label_dir)
            if f.endswith('.png')
        ])

        assert len(self.image_paths) == len(self.label_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"

        print(f"Loaded {len(self.image_paths)} GTA5 images.")


    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx])

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.image_paths)
