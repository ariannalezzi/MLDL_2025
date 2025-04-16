import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.cityscapes import CityScapes
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from train import train_one_epoch, validate

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
root_dir = "/content/cityscapes/Cityscapes/Cityspaces" 
batch_size = 4
epochs = 50
base_lr = 0.01
best_miou = 0.0

# Transforms
transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor()
])

# Dataset & Dataloader
train_dataset = CityScapes(root=root_dir, split='train', transform=transform, target_transform=target_transform)
val_dataset = CityScapes(root=root_dir, split='val', transform=transform, target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# Model
model = get_deeplab_v2(num_classes=num_classes, pretrain=False).to(device)
model.multi_level = False

# Optimizer
optimizer = torch.optim.SGD(model.optim_parameters(base_lr), lr=base_lr, momentum=0.9, weight_decay=5e-4)

# Training Loop
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer, base_lr, epoch, epochs, device)
    best_miou, val_miou, ious = validate(model, val_loader, num_classes, device, best_miou)



import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from datasets.cityscapes import CityScapes
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from utils_p import *
from train import *




def main():
    #Dataset definitions and transformation

        # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    dataset_root = '/content/cityscapes/Cityscapes/Cityspaces' #meant to run on colab

    train_dataset = CityScapes(root=dataset_root, split='train', transform=transform, target_transform=target_transform)
    print(f"Train dataset size: {len(train_dataset)}")

    test_dataset = CityScapes(root=dataset_root, split='val', transform=transform, target_transform=target_transform)
    print(f"Test dataset size: {len(test_dataset)}")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=2)


    #MODEL AND HYPERPARAMETERS
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_classes = 19
    base_lr = 2.5e-4
    batch_size = 2
    epochs = 50
    crop_size = (512, 1024)

    # Model
    model = get_deeplab_v2(num_classes=num_classes, pretrain=True, pretrain_model_path='/content/drive/MyDrive/ML&DL/Progetto/MLDL_2025-main/deeplab_resnet_pretrained_imagenet.pth').to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Assuming it's a segmentation task
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_miou = 0.0

    # training loop
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, base_lr, epoch, epochs, device)

    #validation
    validate(model, test_loader, num_classes, device, best_miou)

    #get statistics
    flops, params = get_model_stats(model, input_shape=(3,512,1024), device = device)
    latency = measure_latency(model, input_shape=(3,512,1024), device = device)

    if __name__ == "main":
        main()