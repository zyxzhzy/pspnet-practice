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

# 测试函数
def test_model(model, test_loader, device):
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load('best_model.pth'))

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs= model(images,use_aux=False)
            preds = torch.argmax(outputs, dim=1)

            # 可视化
            for i in range(images.size(0)):
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                mask = masks[i].cpu().numpy()
                pred = preds[i].cpu().numpy()
    
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(image)
                ax[0].set_title("Image")
                ax[1].imshow(mask, cmap='gray')
                ax[1].set_title("Ground Truth")
                ax[2].imshow(pred, cmap='gray')
                ax[2].set_title("Prediction")
                plt.show()






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = pspnet.PSPNet()


    # 测试
test_model(model, test_loader, device)