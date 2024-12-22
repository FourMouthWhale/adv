import torch
import torch.nn as nn


def NI_FGSM(model, criterion, original_images, labels, epsilon, alpha=0.001, num_iterations=10):
    """
    NI-FGSM (Nesterov Iterative Fast Gradient Sign Method) 

    参数:
    - model: 要攻击的模型，需为继承自nn.Module的类实例，实现了前向传播逻辑
    - criterion: 损失函数，如nn.CrossEntropyLoss等，用于计算模型输出与真实标签间的损失
    - original_images: 原始输入图像数据，形状为 (batch_size, channels, height, width) 的torch.Tensor，值范围通常在[0, 1]
    - labels: 对应的真实标签，形状为 (batch_size,) 的torch.Tensor
    - epsilon: 最大扰动幅度，控制生成对抗样本的扰动程度
    - alpha: 每次迭代的步长，用于更新对抗样本
    - num_iterations: 迭代次数，即执行多少次梯度更新来生成对抗样本
    """
    # 复制原始图像作为初始的对抗样本，并设置其需要计算梯度
    perturbed_image = original_images.clone().detach().requires_grad_(True)

    for _ in range(num_iterations):
        # 计算 "前瞻" 点（基于当前对抗样本和当前梯度方向预估的下一步位置）
        lookahead_image = perturbed_image + alpha * torch.sign(perturbed_image.grad.data) if perturbed_image.grad is not None else perturbed_image
        # 前向传播得到模型输出
        outputs = model(lookahead_image)
        # 计算损失
        loss = criterion(outputs, labels)

        # 清空模型之前的梯度信息
        model.zero_grad()
        # 反向传播计算梯度
        loss.backward()

        # 获取当前梯度数据
        data_grad = lookahead_image.grad.data if lookahead_image.grad is not None else torch.zeros_like(original_images)
        # 计算符号梯度
        sign_data_grad = torch.sign(data_grad)

        # 更新对抗样本
        perturbed_image = perturbed_image + epsilon * sign_data_grad
        # 投影操作，确保扰动后的图像仍在合理范围内（这里假设图像范围是[0, 1]）
        perturbed_image = torch.where(perturbed_image > original_images + epsilon,
                                      original_images + epsilon, perturbed_image)
        perturbed_image = torch.where(perturbed_image < original_images - epsilon,
                                      original_images - epsilon, perturbed_image)
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach().requires_grad_(True)

    return perturbed_image