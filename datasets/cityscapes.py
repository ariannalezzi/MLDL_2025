import os
from PIL import Image
import torch
from torch.utils.data import Dataset

#Custom PyTorch Dataset for loading Cityscapes dataset images and labels
class CityScapes(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Define directories for images and ground truth labels
        self.image_dir = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'gtFine', split)

        # Gather all image-label path pairs
        self.image_paths = []
        self.label_paths = []

        # Traverse each city folder to gather image/label file paths
        for city in os.listdir(self.image_dir):
            city_img_folder = os.path.join(self.image_dir, city)
            city_label_folder = os.path.join(self.label_dir, city)

            for filename in os.listdir(city_img_folder):
                if filename.endswith('leftImg8bit.png'):
                    img_path = os.path.join(city_img_folder, filename)
                    label_filename = filename.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
                    label_path = os.path.join(city_label_folder, label_filename)

                    if os.path.exists(label_path):  # add to list only if label exists (some test data may not have labels)
                        self.image_paths.append(img_path)
                        self.label_paths.append(label_path)

        print(f"Loaded {len(self.image_paths)} images for split: {split}") # for verification/debugging

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




'''
import os 
from torch.utils.data import Dataset
from torchvision.io import read_image # to read images from files and load as tensor

# TODO: implement here your custom dataset class for Cityscapes
# image_dir -> path to the folder containing the images

class CityScapes(Dataset):
    def __init__(self, image_dir, image_mask, transform=None, target_transform=None):
        super(CityScapes, self).__init__()
        self.image_dir = image_dir
        self.image_mask = image_mask
        self.transform = transform
        self.target_transform = target_transform

        #download and read data 
        
        # Collect all image and mask paths
        self.image_paths = []
        self.mask_paths = []

        for root, _, files in os.walk(self.image_dir): # esplora ricorsivamente la directory image_dir per raccogliere i percorsi di tutte le immagini che terminano con _gtFine_color.png.
            for file in files:
                if file.endswith('_gtFine_color.png'): # vuol dire che il file è l'immagine 
                    img_path = os.path.join(root, file)
                    mask_path = img_path.replace('_gtFine_color', '_gtFine_labelIds') # così identifica la mask appropriata sostituendo con la finale della maschera
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx])
        mask = read_image(self.mask_paths[idx])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        sample = {"image": image, "label": mask}
        return sample

    def __len__(self):
        return len(self.image_paths)
    


#Custom PyTorch Dataset for loading Cityscapes dataset images and labels
class CityScapes(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Define directories for images and ground truth labels
        self.image_dir = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'gtFine', split)

        # Gather all image-label path pairs
        self.image_paths = []
        self.label_paths = []

        # Traverse each city folder to gather image/label file paths
        for city in os.listdir(self.image_dir):
            city_img_folder = os.path.join(self.image_dir, city)
            city_label_folder = os.path.join(self.label_dir, city)

            for filename in os.listdir(city_img_folder):
                if filename.endswith('leftImg8bit.png'):
                    img_path = os.path.join(city_img_folder, filename)
                    label_filename = filename.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
                    label_path = os.path.join(city_label_folder, label_filename)

                    if os.path.exists(label_path):  # add to list only if label exists (some test data may not have labels)
                        self.image_paths.append(img_path)
                        self.label_paths.append(label_path)

        print(f"Loaded {len(self.image_paths)} images for split: {split}") # for verification/debugging

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

'''


