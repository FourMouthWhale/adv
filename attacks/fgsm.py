import torch
import torch.nn as nn
from .utils import compute_gradient


def FGSM(model, criterion, original_images, labels, epsilon):
    """
    FGSM (Fast Gradient Sign Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 扰动幅度
    
    """
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    data_grad = compute_gradient(model, criterion, perturbed_images, labels)
    sign_data_grad = data_grad.sign()
    perturbed_images = perturbed_images + epsilon * sign_data_grad
    perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
    
    return perturbed_images