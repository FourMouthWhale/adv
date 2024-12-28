import torch
import torch.nn as nn

def IE_FGSM(model, criterion, original_images, labels, epsilon, num_iterations=10):
    """
    IE-FGSM (Enhanced Euler's Method - Fast Gradient Sign Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    """
    # 计算步长
    alpha = epsilon / num_iterations
    # 复制原始图像作为初始的对抗样本
    perturbed_images = original_images.clone().detach().requires_grad_(True)

    for _ in range(num_iterations):
        # 前向传播
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)

        # 反向传播
        model.zero_grad()
        loss.backward()

        # 获取当前对抗样本的梯度
        data_grad = perturbed_images.grad.data
        # 计算当前梯度的归一化值
        gp = data_grad / torch.sum(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)
        # 计算预期对抗样本的梯度
        perturbed_images_plus_gp = (perturbed_images + gp).detach().requires_grad_(True)
        outputs_plus_gp = model(perturbed_images_plus_gp)
        loss_plus_gp = criterion(outputs_plus_gp, labels)
        model.zero_grad()
        loss_plus_gp.backward()
        ga = perturbed_images_plus_gp.grad.data / torch.sum(torch.abs(perturbed_images_plus_gp.grad.data), dim=(1, 2, 3), keepdim=True)
        # 计算搜索方向
        phi = (gp + ga) / 2
        # 更新对抗样本
        perturbed_images = perturbed_images + alpha * phi.sign()
        # 裁剪对抗样本，使其在原始图像的epsilon范围内
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images