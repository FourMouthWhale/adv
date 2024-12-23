import torch
import torch.nn as nn

def EMI_FGSM(model, criterion, original_images, labels, epsilon, num_iterations=10, decay=1, sampling_num=2, eta=7):
    """
    EMI-FGSM (Enhanced Momentum Iterative Fast Gradient Sign Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像（假设为批量图像，形状为 (batch_size, channels, height, width)）
    - labels: 原始图像的标签（形状为 (batch_size,)）
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - decay: 动量衰减因子
    - sampling_num: 采样数量
    - eta: 采样区间边界
    """
    alpha = epsilon / num_iterations
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(original_images).detach().to(original_images.device)
    prev_avg_grad = torch.zeros_like(original_images).detach().to(original_images.device)

    for _ in range(num_iterations):
        # 采样系数
        c = torch.rand(sampling_num) * 2 * eta - eta
        c = c.view(-1, 1, 1, 1, 1).to(original_images.device)

        # 计算采样数据点，利用广播机制同时处理多个采样点
        sampled_images = perturbed_images + c * prev_avg_grad
        sampled_images = torch.clamp(sampled_images, original_images - epsilon, original_images + epsilon)
        sampled_images = sampled_images.view(-1, *original_images.shape[1:]).clone().detach().requires_grad_(True)
        
        # 计算采样数据点的梯度
        outputs = model(sampled_images)  # 将采样图像展平为 (sampling_num * batch_size, channels, height, width)
        loss = criterion(outputs, labels.repeat(sampling_num))  # 重复标签以匹配输出形状
        model.zero_grad()
        loss.backward()
        grads = sampled_images.grad.view(sampling_num, -1, *original_images.shape[1:])  # 将梯度形状恢复为 (sampling_num, batch_size, channels, height, width)
        sampled_images.grad.zero_()

        # 计算平均梯度
        avg_grad = torch.mean(grads, dim=0)

        # 更新增强动量
        momentum = decay * momentum + avg_grad / torch.sum(torch.abs(avg_grad), dim=(1, 2, 3), keepdim=True)
        sign_data_grad = momentum.sign()

        # 更新对抗样本
        perturbed_images = perturbed_images + alpha * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

        # 更新前一次迭代的平均梯度
        prev_avg_grad = avg_grad

    return perturbed_images