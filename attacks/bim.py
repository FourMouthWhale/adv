import torch
import torch.nn as nn

def BIM(model, criterion, original_images, labels, epsilon, num_iterations=10):
    """
    BIM (Basic Iterative Method)
    I-FGSM (Iterative Fast Gradient Sign Method)

    参数:
    model: 要攻击的模型
    criterion: 损失函数
    original_images: 原始图像
    labels: 原始图像的标签
    epsilon: 最大扰动幅度
    num_iterations: 迭代次数 
    
    """
    # alpha 每次迭代步长
    alpha = epsilon / num_iterations
    perturbed_images = original_images.clone().detach().requires_grad_(True)

    for _ in range(num_iterations):
        # 计算损失
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        # 计算梯度
        loss.backward()

        # 更新对抗样本
        perturbation = alpha * perturbed_images.grad.sign()
        perturbed_images = perturbed_images + perturbation
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images