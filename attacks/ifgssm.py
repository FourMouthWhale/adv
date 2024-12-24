import torch
import torch.nn as nn

def I_FGSSM(model, criterion, original_images, labels, epsilon, num_iterations=10, staircase_num=64):
    """
    I-FGSSM (Iterative Fast Gradient Staircase Sign Method)

    参数:
    model: 要攻击的模型
    criterion: 损失函数
    original_images: 原始图像
    labels: 原始图像的标签
    epsilon: 最大扰动幅度
    num_iterations: 迭代次数
    staircase_num: 阶梯数量（默认为64）
    """
    # 计算步长alpha
    alpha = epsilon / num_iterations
    perturbed_images = original_images.clone().detach().requires_grad_(True)

    for _ in range(num_iterations):
        # 计算损失
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        # 计算梯度
        loss.backward()
        gradients = perturbed_images.grad

        # 计算梯度绝对值的百分位数
        percentile_interval = 100 / staircase_num
        percentiles = []
        for p in [percentile_interval * i for i in range(1, staircase_num + 1)]:
            percentiles.append(torch.quantile(gradients.abs().view(-1), p / 100))

        # 根据百分位数分配阶梯权重
        weights = torch.zeros_like(gradients)
        for i in range(staircase_num):
            lower_bound = percentiles[i - 1] if i > 0 else 0
            upper_bound = percentiles[i]
            mask = (lower_bound <= gradients.abs()) & (gradients.abs() <= upper_bound)
            weights[mask] = (2 * i + 1) * percentile_interval / 100

        # 更新对抗样本
        perturbation = alpha * gradients.sign() * weights
        perturbed_images = perturbed_images + perturbation
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images