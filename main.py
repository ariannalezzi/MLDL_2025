import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.cityscapes import CityScapes
from models.deepLabv2.deeplabv2 import get_deeplab_v2
from train import train_one_epoch, validate

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
root_dir = "/content/dataset"  # update path if different
num_classes = 19
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

# Optimizer
optimizer = torch.optim.SGD(model.optim_parameters(base_lr), lr=base_lr, momentum=0.9, weight_decay=5e-4)

# Training Loop
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer, base_lr, epoch, epochs, device)
    best_miou, val_miou, ious = validate(model, val_loader, num_classes, device, best_miou)
