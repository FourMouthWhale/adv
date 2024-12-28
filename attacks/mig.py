import torch
import torch.nn as nn

def MIG(model, original_images, labels, epsilon, num_iterations=25, decay=1, baseline_images=None):
    """
    MIG (Momentum Integrated Gradients)算法

    参数:
    - model: 要攻击的模型
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - decay: 动量衰减因子
    - baseline_images: 基线图像，默认为黑色图像（全零张量）
    """
    if baseline_images is None:
        baseline_images = torch.zeros_like(original_images)

    alpha = epsilon / num_iterations
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(original_images).detach().to(original_images.device)

    for _ in range(num_iterations):
        # 计算积分梯度
        integrated_gradients = calculate_integrated_gradients(model, perturbed_images, baseline_images)
        # 更新动量
        momentum = decay * momentum + integrated_gradients / torch.sum(torch.abs(integrated_gradients), dim=(1, 2, 3), keepdim=True)
        sign_data_grad = momentum.sign()

        # 更新对抗样本
        perturbed_images = perturbed_images + alpha * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images

def calculate_integrated_gradients(model, input_images, baseline_images, num_steps=20):
    """
    计算积分梯度

    参数:
    - model: 模型
    - input_images: 输入图像
    - baseline_images: 基线图像
    - num_steps: 计算积分梯度的步数
    """
    integrated_gradients = torch.zeros_like(input_images)

    for step in range(num_steps):
        step_size = 1.0 / num_steps
        interpolated_images = baseline_images + step_size * (input_images - baseline_images)
        interpolated_images.requires_grad_(True)
        outputs = model(interpolated_images)
        model.zero_grad()
        outputs.backward(torch.ones_like(outputs))
        gradients = interpolated_images.grad.data
        integrated_gradients += gradients * (input_images - baseline_images) * step_size

    return integrated_gradients