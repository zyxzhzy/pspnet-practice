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





# def get_transforms():
#     return transforms.Compose([
#         transforms.ToTensor(),  # 将图像转换为张量并归一化到[0, 1]
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
#     ])

class VOCSegmentationDataset(Dataset):
    def __init__(self, 
                 root_dir='/data/zhxie/VOCdevkit/VOC2012', 
                 split='train', 
                 crop_size=224, 
                 scale_range=(0.5, 2.0), 
                 use_aug=True, 
                 is_train=True):
        """
        Args:
            root_dir (str): VOC数据集根目录
            split (str): 数据集划分（'train', 'val', 'trainval'）
            crop_size (int): 训练时随机裁剪的尺寸（PSPNet常用473）
            scale_range (tuple): 随机缩放比例范围
            use_aug (bool): 是否使用增强标注（SegmentationClassAug）
            is_train (bool): 训练模式（启用数据增强）
        """
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.is_train = is_train
        self.use_aug = use_aug
        
        # 路径配置
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClassAug' if use_aug else 'SegmentationClass')
        self.split_file = os.path.join(root_dir, 'ImageSets/Segmentation', f'{split}.txt')
        
        # 加载文件名列表
        with open(self.split_file, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]
        
        # 数据标准化参数（ImageNet统计值）
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.filenames)
    
    def _random_scale_crop(self, img, mask):
        """ 修复尺寸计算的缩放裁剪 """
        # 计算最小允许缩放比例
        orig_h, orig_w = img.height, img.width
        min_scale_h = self.crop_size / orig_h
        min_scale_w = self.crop_size / orig_w
        min_scale = max(min_scale_h, min_scale_w, self.scale_range[0])
    
        # 生成有效缩放比例
        scale = np.random.uniform(min_scale, self.scale_range[1])
    
        # 计算缩放后尺寸
        scaled_h = int(orig_h * scale)
        scaled_w = int(orig_w * scale)
    
        # 双线性插值缩放图像
        img = TF.resize(img, (scaled_h, scaled_w), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (scaled_h, scaled_w), interpolation=Image.NEAREST)
    
        # 随机裁剪
        i, j, h, w = transforms.RandomCrop.get_params(
            img, 
            output_size=(self.crop_size, self.crop_size)
        )
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
    
        return img, mask
    
    def _train_transform(self, img, mask):
        """ 训练数据增强 """
        # 随机水平翻转
        if np.random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        
        # 随机缩放 + 裁剪
        img, mask = self._random_scale_crop(img, mask)

        
        # 颜色抖动（仅图像）
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        img = color_jitter(img)
        
        # 转换为Tensor并标准化
        img = TF.to_tensor(img)
        img = self.normalize(img)
        mask = torch.from_numpy(np.array(mask)).long()
        
        return img, mask
    
    def _val_transform(self, img, mask):
        """ 验证集预处理 """
        # 中心裁剪或调整尺寸
        img = TF.resize(img, (self.crop_size,self.crop_size), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (self.crop_size,self.crop_size),  interpolation=Image.NEAREST)
        
        img = TF.to_tensor(img)
        img = self.normalize(img)
        mask = torch.from_numpy(np.array(mask)).long()
        
        return img, mask
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.img_dir, filename + '.jpg')
        mask_path = os.path.join(self.mask_dir, filename + '.png')
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # 忽略边界标签（根据VOC官方设置）
        mask = np.array(mask)
        mask[mask == 255] = 0  # 将255（边界）设为0（背景）
        mask = Image.fromarray(mask)
        
        if self.is_train:
            img, mask = self._train_transform(img, mask)
        else:
            img, mask = self._val_transform(img, mask)
        print(filename)
        print(mask)
        return img, mask




# class DilatedResNet(nn.Module):
#     def __init__(self, backbone='resnet50', pretrained=True):
#         super().__init__()
#         resnet = getattr(torchvision.models, backbone)(pretrained=pretrained)
        
#         # 移除全连接层和平均池化
#         del resnet.avgpool
#         del resnet.fc

#         # 修改layer3和layer4的空洞卷积参数
#         self._modify_layer(resnet.layer3, dilation=2)
#         self._modify_layer(resnet.layer4, dilation=4)
        
#         # 分解各层
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4

#     def _modify_layer(self, layer, dilation):
#         """调整空洞卷积参数"""
#         for block in layer.children():
#             if isinstance(block, torchvision.models.resnet.Bottleneck):
#                 if block.conv2.stride == (2, 2):
#                     block.conv2.stride = (1, 1)
#                     block.conv2.dilation = (dilation, dilation)
#                     block.conv2.padding = (dilation, dilation)
#                     if block.downsample is not None:
#                         block.downsample[0].stride = (1, 1)
#                         # 添加空洞卷积到下采样层（可选）
#                         #block.downsample[0].dilation = (dilation, dilation)
#                         block.downsample[0].padding = (dilation, dilation)

#     def forward(self, x):
#         print("Input:", x.shape)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         print("After maxpool:", x.shape)

#         x = self.layer1(x)
#         print("After layer1:", x.shape)
#         x = self.layer2(x)
#         print("After layer2:", x.shape)
#         aux_feat = self.layer3(x)  # 辅助特征
#         print("After layer3:", aux_feat.shape)
#         main_feat = self.layer4(aux_feat)  # 主特征
#         print("After layer4:", main_feat.shape)
#         return main_feat, aux_feat


    
# class PSPModule(nn.Module):
#     """金字塔池化模块（保持原始设计）"""
#     def __init__(self, in_channels, bin_sizes=(1, 2, 3, 6)):
#         super().__init__()
#         self.branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.AdaptiveAvgPool2d(size),
#                 nn.Conv2d(in_channels, in_channels//4, 1, bias=False),
#                 nn.BatchNorm2d(in_channels//4),
#                 nn.ReLU(inplace=True))
#             for size in bin_sizes
#         ])
        
#         self.fusion = nn.Sequential(
#             nn.Conv2d(in_channels*2, 512, 3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         h, w = x.size()[2:]
#         features = [x]
#         for branch in self.branches:
#             out = branch(x)
#             out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
#             features.append(out)
#         return self.fusion(torch.cat(features, dim=1))

# class PSPNet(nn.Module):
#     """带辅助损失头的完整网络"""
#     def __init__(self, num_classes=21, backbone='resnet50'):  # ADE20K有150个类别
#         super().__init__()
        
#         # 骨干网络
#         self.backbone = DilatedResNet(backbone)
        
#         # 通道数配置
#         backbone_channels = {
#             'resnet50': {'main': 2048, 'aux': 1024},
#             'resnet101': {'main': 2048, 'aux': 1024},
#             'resnet18': {'main': 512, 'aux': 256},
#             'resnet34': {'main': 512, 'aux': 256}
#         }
#         channels = backbone_channels[backbone]
        
#         # 主网络分支
#         self.psp = PSPModule(channels['main'])
#         self.main_classifier = nn.Sequential(
#             nn.Conv2d(512, num_classes, 1),
#             nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
#         )
        
#         # 辅助损失分支
#         self.aux_classifier = nn.Sequential(
#             nn.Conv2d(channels['aux'], 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(256, num_classes, 1),
#             nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
#         )

#     def forward(self, x,use_aux=True):
#         # 前向传播
#         main_feat, aux_feat = self.backbone(x)
        
#         # 主分支
#         psp_out = self.psp(main_feat)
#         main_out = self.main_classifier(psp_out)
        
#         if use_aux:
#             aux_out = self.aux_classifier(aux_feat)
#             return main_out, aux_out
#         else:
#             return main_out





