import os
import math
import torch
import argparse
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, random_split
from utils.tool import *
from utils.datasets import *
from module.loss import DetectorLoss, CombinedLoss
from trafficlight_cls import LightClassifier
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError

class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__(root, transform=None)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            image = self.loader(path)
        except (OSError, UnidentifiedImageError) as e:
            print(f"Warning: Skipping corrupted image: {path}")
            return None, None

        if self.albumentations_transform is not None:
            image = self.albumentations_transform(image=np.array(image))['image']

        return image, target

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer_model, optimizer_center, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_cl_loss = 0.0

        for inputs, labels in train_dataloader:
            if inputs is None:  # Skip batches with None
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_model.zero_grad()
            optimizer_center.zero_grad()

            features, logits = model(inputs)
            loss, ce_loss, cl_loss = criterion(features, labels, logits)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(criterion.center_loss.parameters(), max_norm=1.0)

            loss.backward()
            optimizer_model.step()
            optimizer_center.step()

            running_loss += loss.item() * inputs.size(0)
            running_ce_loss += ce_loss.item() * inputs.size(0)
            running_cl_loss += cl_loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_ce_loss = running_ce_loss / len(train_dataloader.dataset)
        epoch_cl_loss = running_cl_loss / len(train_dataloader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, '
              f'Loss: {epoch_loss:.4f}, '
              f'CE Loss: {epoch_ce_loss:.4f}, '
              f'Center Loss: {epoch_cl_loss:.4f}')

        if epoch % 10 == 0 and epoch > 0:
            model.eval()
            val_running_loss = 0.0
            val_running_ce_loss = 0.0
            val_running_cl_loss = 0.0
            correct = 0
            total = 0
            torch.save(model.state_dict(), "weights/tflt/tflt_weight_loss:%f_%d-epoch.pth" % (epoch_loss, epoch))

            with torch.no_grad():
                for val_inputs, val_labels in val_dataloader:
                    if val_inputs is None:  # Skip batches with None
                        continue
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)

                    val_features, val_logits = model(val_inputs)
                    val_loss, val_ce_loss, val_cl_loss = criterion(val_features, val_labels, val_logits)

                    val_running_loss += val_loss.item() * val_inputs.size(0)
                    val_running_ce_loss += val_ce_loss.item() * val_inputs.size(0)
                    val_running_cl_loss += val_cl_loss.item() * val_inputs.size(0)
                    
                    _, predicted = torch.max(val_logits, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()

            val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
            val_epoch_ce_loss = val_running_ce_loss / len(val_dataloader.dataset)
            val_epoch_cl_loss = val_running_cl_loss / len(val_dataloader.dataset)
            val_accuracy = correct / total

            print(f'Validation - Epoch {epoch}/{num_epochs - 1}, '
                  f'Val Loss: {val_epoch_loss:.4f}, '
                  f'Val CE Loss: {val_epoch_ce_loss:.4f}, '
                  f'Val Center Loss: {val_epoch_cl_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')

if __name__ == "__main__":
    data_dir = '/home/achintya-trn0175/Downloads/RainWeatherClassification/newtrainingfinalimages'

    data_transforms = {
        'train': A.Compose([
            A.Resize(200, 200),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.JpegCompression(quality_lower=70, quality_upper=90, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'val': A.Compose([
            A.Resize(200, 200),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
    }

    full_dataset = CustomDataset(root=data_dir, transform=data_transforms['train'])

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.albumentations_transform = data_transforms['val']

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    class_names = full_dataset.classes
    print("Classes:", class_names)
    
    model = LightClassifier(len(class_names), False).to(device)

    lossfunc = CombinedLoss(num_classes=len(class_names), feat_dim=192, device=device)
    
    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.949, weight_decay=0.0005)
    optimizer_center = torch.optim.SGD(lossfunc.center_loss.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, lossfunc, optimizer, optimizer_center, num_epochs=201)
