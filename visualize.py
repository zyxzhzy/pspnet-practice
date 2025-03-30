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
# PASCAL VOC 21类（含背景）的官方颜色映射
VOC_COLORMAP = [
    [0, 0, 0],        # 背景（0）
    [128, 0, 0],       # 飞机（1）
    [0, 128, 0],       # 自行车（2）
    [128, 128, 0],     # 鸟（3）
    [0, 0, 128],       # 船（4）
    [128, 0, 128],     # 瓶子（5）
    [0, 128, 128],     # 公交车（6）
    [128, 128, 128],   # 汽车（7）
    [64, 0, 0],        # 猫（8）
    [192, 0, 0],       # 椅子（9）
    [64, 128, 0],      # 牛（10）
    [192, 128, 0],     # 餐桌（11）
    [64, 0, 128],      # 狗（12）
    [192, 0, 128],     # 马（13）
    [64, 128, 128],    # 摩托车（14）
    [192, 128, 128],   # 人（15）
    [0, 64, 0],        # 盆栽（16）
    [128, 64, 0],      # 羊（17）
    [0, 192, 0],       # 沙发（18）
    [128, 192, 0],     # 火车（19）
    [0, 64, 128]       # 显示器（20）
]

# 将颜色表转换为NumPy数组（方便索引）
voc_colormap = np.array(VOC_COLORMAP, dtype=np.uint8)


def decode_segmap(pred_mask, colormap=voc_colormap):
    """
    将预测的类别索引矩阵转换为RGB颜色图像
    Args:
        pred_mask (np.ndarray): [H, W] 值为0-20的整数矩阵
        colormap (np.ndarray): [num_classes, 3] 的颜色映射表
    Returns:
        np.ndarray: [H, W, 3] 的RGB图像
    """
    # 创建空RGB图像
    rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    
    # 为每个类别索引填充颜色
    for cls in range(colormap.shape[0]):
        rgb[pred_mask == cls] = colormap[cls]
    
    return rgb



def visualize_comparison(image, true_mask, pred_mask, save_path=None):
    """
    并排显示原始图像、真值标注和预测结果
    Args:
        image (PIL.Image): 原始RGB图像
        true_mask (np.ndarray): 真值标注矩阵 [H, W]
        pred_mask (np.ndarray): 预测结果矩阵 [H, W]
        save_path (str): 若提供，保存图像到此路径
    """
    # 转换预测结果为颜色图
    pred_rgb = decode_segmap(pred_mask)
    true_rgb = decode_segmap(true_mask)
    
    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示真值标注
    axes[1].imshow(true_rgb)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # 显示预测结果
    axes[2].imshow(pred_rgb)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"可视化结果已保存至 {save_path}")
    else:
        plt.show()
    plt.close()


def predict_and_visualize(image_path, model, device='cuda', save_path=None):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # 预处理（与训练保持一致）
    transform = transforms.Compose([
        transforms.Resize(473),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 473, 473]
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()  # [473, 473]
    
    # 将预测结果还原到原始图像尺寸
    pred_mask = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask = pred_mask.resize(original_size, Image.NEAREST)  # 保持类别索引
    pred_mask = np.array(pred_mask)
    
    # 如果有真值标注，加载并处理（示例路径处理逻辑）
    # true_mask_path = image_path.replace('JPEGImages', 'SegmentationClass').replace('.jpg', '.png')
    # true_mask = np.array(Image.open(true_mask_path))
    # true_mask = (true_mask >= 0) & (true_mask <= 20)  # 处理可能的255边界
    
    # 可视化（此处假设没有真值，仅显示预测）
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Input')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(decode_segmap(pred_mask))
    plt.title('Prediction')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 使用示例
model = pspnet.PSPNet().to(device)
model.load_state_dict(torch.load('best_model.pth'))

predict_and_visualize(
    image_path='/home/zhxie/pspnet/VOC2012/JPEGImages/2012_004312.jpg',
    model=model,
    save_path='prediction_visualization.png'
)
