import torch
import torch.nn as nn

def GNP(model, criterion, original_images, labels, epsilon, num_iterations=10, step_size=0.01, beta=0.8):
    """
    I-FGSM with Gradient Norm Penalty (GNP)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - step_size: 步长
    - beta: 正则化系数
    """
    # 计算每次迭代的步长
    alpha = epsilon / num_iterations
    # 复制原始图像作为初始的对抗样本
    perturbed_images = original_images.clone().detach().requires_grad_(True)

    for _ in range(num_iterations):
        # 计算原始损失函数的梯度
        data_grad = compute_gradient(model, criterion, perturbed_images, labels)

        # 计算用于近似GNP项中梯度范数惩罚的梯度
        g2_images = perturbed_images + step_size * (data_grad / data_grad.norm(p=2, dim=(1, 2, 3), keepdim=True))
        g2_images = g2_images.detach().requires_grad_(True)
        g2_grad = compute_gradient(model, criterion, g2_images, labels)

        # 更新梯度
        new_grad = (1 + beta) * data_grad - beta * g2_grad
        # 计算符号梯度
        sign_data_grad = new_grad.sign()

        # 更新对抗样本
        perturbed_images = perturbed_images + alpha * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images

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