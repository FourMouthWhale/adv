import torch
import torch.nn as nn


def MI_FGSM(model, criterion, original_images, labels, epsilon, alpha=0.001, num_iterations=10, decay=1):
    """
    MI-FGSM (Momentum Iterative Fast Gradient Sign Method) 

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - alpha: 每次迭代的步长
    - num_iterations: 迭代次数
    - decay: 动量衰减因子
    """
    # 复制原始图像作为初始的对抗样本
    perturbed_image = original_images.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(original_images).detach().to(original_images.device)

    for _ in range(num_iterations):
        outputs = model(perturbed_image)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = perturbed_image.grad.data
        # 归一化梯度，避免梯度爆炸等问题
        data_grad = data_grad / torch.mean(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)
        # 更新动量
        momentum = decay * momentum + data_grad / torch.mean(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)
        # 计算带动量的符号梯度
        sign_data_grad = momentum.sign()

        # 更新对抗样本
        perturbed_image = perturbed_image + alpha * sign_data_grad
        # 投影操作，确保扰动后的图像仍在合理范围内（这里假设图像范围是[0, 1]）
        perturbed_image = torch.where(perturbed_image > original_images + epsilon,
                                      original_images + epsilon, perturbed_image)
        perturbed_image = torch.where(perturbed_image < original_images - epsilon,
                                      original_images - epsilon, perturbed_image)
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach().requires_grad_(True)

    return perturbed_image