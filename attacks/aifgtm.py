import torch
import torch.nn as nn
from math import sqrt

def AI_FGTM(model, criterion, original_images, labels, epsilon, num_iterations=10, beta1=0.9, beta2=0.999, mu1=0.9, mu2=0.999, lambda_=1.0):
    """
    AI-FGTM (Adam Iterative Fast Gradient Tanh Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - beta1: Adam算法中的第一指数衰减率
    - beta2: Adam算法中的第二指数衰减率
    - mu1: 第一时刻因子
    - mu2: 第二时刻因子
    - lambda_: 尺度因子
    """
    # 初始化对抗样本为原始图像
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    m = torch.zeros_like(original_images).detach().to(original_images.device)
    v = torch.zeros_like(original_images).detach().to(original_images.device)

    for t in range(num_iterations):
        # 计算当前步长
        step_size = epsilon * (1 - beta1 ** (t + 1)) / (sqrt(1 - beta2 ** (t + 1))) * sum((1 - beta1 ** (i + 1)) / sqrt(1 - beta2 ** (i + 1)) for i in range(num_iterations))

        # 前向传播
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = perturbed_images.grad.data

        # 更新一阶矩m
        m = mu1 * m + (1 - mu1) * data_grad
        # 更新二阶矩v
        v = mu2 * v + (1 - mu2) * data_grad ** 2

        # 使用tanh函数计算更新方向
        update_direction = torch.tanh(lambda_ * m / (torch.sqrt(v) + 1e-8))

        # 更新对抗样本
        perturbed_images = perturbed_images + step_size * update_direction
        # 裁剪对抗样本，使其在原始图像的epsilon范围内
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images