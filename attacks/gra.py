import torch
import torch.nn as nn


def GRA(model, criterion, original_images, labels, epsilon, num_iterations=10, decay=1, beta=3.5, eta=0.94, sample_num=1):
    """
    GRA (Gradient Relevance Attack)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - decay: 动量衰减因子
    - beta: 样本范围上限因子
    - eta: 衰减因子
    - sample_num: 样本数量
    """
    alpha = epsilon / num_iterations
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(original_images).detach().to(original_images.device)
    # 初始化衰减指标M为1/eta
    M = torch.ones_like(original_images).detach().to(original_images.device) / eta

    for _ in range(num_iterations):
        # 计算当前输入的梯度
        current_grad = compute_gradient(model, criterion, perturbed_images, labels)

        # 采样附近图像并计算平均梯度
        sampled_gradients = []
        x_i = perturbed_images.clone()
        for _ in range(sample_num):
            xi = x_i + torch.randn_like(x_i) * (beta * epsilon)
            sampled_gradients.append(compute_gradient(model, criterion, xi, labels))
            x_i = xi
        # 计算采样平均梯度
        average_grad = torch.mean(torch.stack(sampled_gradients), dim=0)

        # 计算余弦相似度
        cosine_similarity = torch.sum(current_grad * average_grad, dim=(1, 2, 3), keepdim=True) / (
                    torch.norm(current_grad, p=2, dim=(1, 2, 3), keepdim=True) * torch.norm(average_grad, p=2, dim=(1, 2, 3), keepdim=True))

        # 计算全局加权梯度
        weighted_grad = cosine_similarity * current_grad + (1 - cosine_similarity) * average_grad

        # 更新动量积累
        old_momentum = momentum.clone().detach()
        momentum = decay * momentum + weighted_grad / torch.sum(torch.abs(weighted_grad), dim=(1, 2, 3), keepdim=True)

        # 更新衰减指标M
        sign_data_grad = momentum.sign()
        M = M * ((sign_data_grad == torch.sign(old_momentum)) + eta * (sign_data_grad!= torch.sign(old_momentum)))

        # 更新对抗样本
        perturbed_images = perturbed_images + alpha * M * sign_data_grad
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