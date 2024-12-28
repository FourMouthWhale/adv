import torch
import torch.nn as nn


def SMI_FGRM(model, criterion, original_images, labels, epsilon, num_iterations=10, decay=1, sampling_num=12, sampling_beta=1.5, rescale_c=2):
    """
    SMI-FGRM (Sampling-based Momentum Iterative Fast Gradient Rescaling Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - decay: 动量衰减因子
    - sampling_num: 采样数量
    - sampling_beta: 采样范围参数
    - rescale_c: 重缩放因子
    """
    alpha = epsilon / num_iterations
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(original_images).detach().to(original_images.device)

    for _ in range(num_iterations):
        # 深度优先采样
        sampled_gradients = []
        x_i = perturbed_images.clone()
        for _ in range(sampling_num):
            xi = x_i + torch.randn_like(x_i) * (sampling_beta * epsilon)
            sampled_gradients.append(compute_gradient(model, criterion, xi, labels))
            x_i = xi
        # 加上原始图像梯度
        sampled_gradients.append(compute_gradient(model, criterion, perturbed_images, labels))
        g_hat = torch.mean(torch.stack(sampled_gradients), dim=0)

        # 更新动量
        momentum = decay * momentum + g_hat / torch.sum(torch.abs(g_hat), dim=(1, 2, 3), keepdim=True)

        # 快速梯度重缩放
        rescaled_gradient = rescale_gradient(momentum, rescale_c)

        # 更新对抗样本
        perturbed_images = perturbed_images + alpha * rescaled_gradient
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images


def rescale_gradient(g, c):
    """
    梯度重缩放函数

    参数:
    - g: 梯度
    - c: 重缩放因子
    """
    normed_log_gradient = (torch.log2(torch.abs(g)) - torch.mean(torch.log2(torch.abs(g)), dim=(1, 2, 3), keepdim=True)) / torch.std(torch.log2(torch.abs(g)), dim=(1, 2, 3), keepdim=True)
    sigmoid_applied = 1 / (1 + torch.exp(-normed_log_gradient))
    return c * torch.sign(g) * sigmoid_applied


def compute_gradient(model, criterion, x, labels):
    """
    计算梯度

    参数:
    - model: 模型
    - criterion: 损失函数
    - x: 输入图像
    - labels: 标签
    """
    x = x.clone().detach().requires_grad_(True)
    outputs = model(x)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    return x.grad.data