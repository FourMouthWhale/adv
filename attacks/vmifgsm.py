import torch
import torch.nn as nn


def VMI_FGSM(model, criterion, original_images, labels, epsilon, num_iterations=10, decay=1, beta=1.5, N=0):
    """
    VMI-FGSM (Variance Tuning Momentum Iterative Fast Gradient Sign Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - decay: 动量衰减因子
    - beta: 邻域因子，用于确定计算梯度方差的邻域范围
    - N: 在邻域内采样的样本数量，用于近似计算梯度方差
    """
    # alpha每次迭代步长
    alpha = epsilon / num_iterations
    # 复制原始图像作为初始的对抗样本
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(original_images).detach().to(original_images.device)

    for _ in range(num_iterations):
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = perturbed_images.grad.data
        # 计算邻域内的梯度方差
        if N > 0:
            variance = variance_tuning(model, criterion, perturbed_images, labels, epsilon, beta, N, data_grad)
        else:
            variance = torch.zeros_like(perturbed_images).detach().to(perturbed_images.device)
        # 更新动量，结合梯度方差
        momentum = decay * momentum + (data_grad + variance) / torch.sum(torch.abs(data_grad + variance), dim=(1, 2, 3), keepdim=True)
        # 计算带动量和方差调整的符号梯度
        sign_data_grad = momentum.sign()

        # 更新对抗样本
        perturbed_images = perturbed_images + alpha * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images


def variance_tuning(model, criterion, perturbed_images, labels, epsilon, beta, N, data_grad):
    """
    计算给定图像的梯度方差

    参数:
    - original_images: 原始图像
    - perturbed_images: 当前的对抗样本
    - model: 要攻击的模型
    - criterion: 损失函数
    - beta: 邻域因子，用于确定计算梯度方差的邻域范围
    - N: 在邻域内采样的样本数量，用于近似计算梯度方差
    """
    epsilon_prime = beta * epsilon
    variance = torch.zeros_like(perturbed_images).detach().to(perturbed_images.device)

    for _ in range(N):
        # 在邻域内随机采样扰动
        random_perturbation = torch.randn_like(perturbed_images).uniform_(-epsilon_prime, epsilon_prime)
        # 应用扰动得到邻域内的样本
        neighbor_images = perturbed_images + random_perturbation
        neighbor_images = torch.clamp(neighbor_images, 0, 1)
        neighbor_images = neighbor_images.detach().requires_grad_(True)

        # 计算邻域样本的梯度
        outputs = model(neighbor_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        neighbor_grad = neighbor_images.grad.data

        # 累加梯度差
        variance += neighbor_grad - data_grad

    # 平均梯度差得到梯度方差
    variance /= N

    return variance