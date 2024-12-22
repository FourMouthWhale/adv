import torch
import torch.nn as nn


def NI_FGSM(model, criterion, original_images, labels, epsilon, num_iterations=10):
    """
    NI-FGSM (Nesterov Iterative Fast Gradient Sign Method) 

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始输入图像数据
    - labels: 对应的真实标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    """
    # alpha: 每次迭代的步长
    alpha = epsilon / num_iterations
    # 复制原始图像作为初始的对抗样本，并设置其需要计算梯度
    perturbed_images = original_images.clone().detach().requires_grad_(True)

    for _ in range(num_iterations):
        # 计算 "前瞻" 点（基于当前对抗样本和当前梯度方向预估的下一步位置）
        lookahead_images = perturbed_images + alpha * torch.sign(perturbed_images.grad.data) if perturbed_images.grad is not None else perturbed_images
        # 前向传播得到模型输出
        outputs = model(lookahead_images)
        # 计算损失
        loss = criterion(outputs, labels)

        # 清空模型之前的梯度信息
        model.zero_grad()
        # 反向传播计算梯度
        loss.backward()

        # 获取当前梯度数据
        data_grad = lookahead_images.grad.data if lookahead_images.grad is not None else torch.zeros_like(original_images)
        # 计算符号梯度
        sign_data_grad = torch.sign(data_grad)

        # 更新对抗样本
        perturbed_images = perturbed_images + epsilon * sign_data_grad
        # 投影操作，确保扰动后的图像仍在合理范围内（这里假设图像范围是[0, 1]）
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)
    
    return perturbed_images