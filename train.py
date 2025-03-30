import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import pspnet
import dataloader

data_dir = '/data/zhxie/VOCdevkit/VOC2012'

batch_size = 4
epochs = 200
learning_rate = 1e-4                                                           
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_dataset =dataloader.VOCSegmentationDataset(
    root_dir='/data/zhxie/VOCdevkit/VOC2012',
    split='train',
    is_train=True,
    use_aug=False
)

val_dataset = dataloader.VOCSegmentationDataset(
    split='val',
    is_train=False,
    use_aug=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)


def calculate_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls).int()
        target_mask = (target == cls).int()
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        ious.append(iou)
    return ious


def calculate_f1(pred, target, num_classes):
    f1s = []
    for cls in range(num_classes):
        pred_mask = (pred == cls).int()
        target_mask = (target == cls).int()
        tp = (pred_mask & target_mask).sum().item()
        fp = (pred_mask & ~target_mask).sum().item()
        fn = (~pred_mask & target_mask).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        f1s.append(f1)
    return f1s


#model = pspnet.PSPNet(num_classes=21)
model = pspnet.PSPNet()
print(model)
def train_model(model, train_loader, val_loader, device, epochs=100, learning_rate=1e-4):
    #model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    aux_criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs, aux_outputs = model(images)
            print( aux_outputs.size())
            print(masks.size())
            loss = criterion(outputs, masks)
            aux_loss = aux_criterion(aux_outputs, masks)
            total_loss = loss + 0.4 * aux_loss  # 权重可以根据需要调整
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

        #验证
        
        if (epoch) % 10 == 0:
            model.eval()
            
            val_loss = 0.0
            val_iou = 0.0  # 初始化 val_iou
            val_f1 = 0.0   # 初始化 val_f1
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    
                    total_loss = loss
                    val_loss += total_loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    ious = calculate_iou(preds, masks, num_classes=21)
                    f1s = calculate_f1(preds, masks, num_classes=21)

                    val_iou += sum(ious) / len(ious)
                    val_f1 += sum(f1s) / len(f1s)

            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            val_f1 /= len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val F1: {val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print("Model saved with best validation loss.")



train_model(model, train_loader, val_loader, device, epochs=epochs, learning_rate=learning_rate)