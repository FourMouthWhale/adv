import torch
import torch.nn as nn

def PI_FGSM(model, criterion, original_images, labels, epsilon, beta=5, kernel_size=3, num_iterations=10):
    """
    PI-FGSM (Patch-wise Iterative Fast Gradient Sign Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 扰动幅度
    - beta: 放大因子
    - kernel_size: 投影核大小
    - num_iterations: 迭代次数
    
    返回:
    - perturbed_image: 生成的对抗样本
    """
    # gamma: 投影因子
    gamma = epsilon / num_iterations * beta
    # 初始化累积放大噪声和裁剪噪声
    a = torch.zeros_like(original_images)
    C = torch.zeros_like(original_images)
    perturbed_images = original_images.clone().detach().requires_grad_(True)

    # 定义投影核
    Wp = torch.ones((kernel_size, kernel_size), dtype=torch.float32) / (kernel_size ** 2 - 1)
    Wp[kernel_size // 2, kernel_size // 2] = 0
    Wp = Wp.expand(original_images.size(1), -1, -1).to(original_images.device)
    Wp = Wp.unsqueeze(0)

    for _ in range(num_iterations):
        # 计算梯度
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = perturbed_images.grad.data

        # 更新累积放大噪声
        a = a + beta * (epsilon / num_iterations) * data_grad.sign()

        # 裁剪噪声
        if a.abs().max() >= epsilon:
            C = (a.abs() - epsilon).clamp(0, float('inf')) * a.sign()
            a = a + gamma * torch.nn.functional.conv2d(input=C, weight=Wp, stride=1, padding=kernel_size // 2)

        # 更新对抗样本
        perturbed_images = perturbed_images + beta * (epsilon / num_iterations) * data_grad.sign() + gamma * torch.nn.functional.conv2d(C, Wp, stride=1, padding=kernel_size // 2)
        
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images