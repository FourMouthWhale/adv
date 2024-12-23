import torch
import torch.nn as nn


def NI_FGSM(model, criterion, original_images, labels, epsilon, num_iterations=10, decay=1):
    """
    NI-FGSM (Nesterov Iterative Fast Gradient Sign Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - decay: 动量衰减因子
    """
    # alpha 每次迭代步长
    alpha = epsilon / num_iterations
    # 复制原始图像作为初始的对抗样本
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(original_images).detach().to(original_images.device)

    for _ in range(num_iterations):
        # 先沿先前累积梯度的方向进行跳跃
        nes_images = perturbed_images + alpha * decay * momentum
        nes_images = nes_images.clone().detach().requires_grad_(True)

        outputs = model(nes_images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = nes_images.grad.data
        # 更新动量 (batch_size, channels, height, width)
        momentum = decay * momentum + data_grad / torch.sum(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)
        # 计算带动量的符号梯度
        sign_data_grad = momentum.sign()

        # 更新对抗样本
        perturbed_images = perturbed_images + alpha * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images