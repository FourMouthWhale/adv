import torch
import torch.nn as nn

def RAP(model, criterion, original_images, labels, epsilon, num_iterations=10, decay=1, epsilon_n=16/255, late_start=5):
    """
    RAP (Reverse Adversarial Perturbation)算法实现

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 总迭代次数
    - decay: 动量衰减因子（在内部优化中未使用，可根据需要调整）
    - epsilon_n: 反向扰动的搜索区域
    - late_start: 晚启动迭代次数
    """
    # 初始化对抗样本为原始图像
    perturbed_images = original_images.clone().detach().requires_grad_(True)

    for i in range(num_iterations):
        if i >= late_start:
            # 初始化反向扰动
            n_rap = torch.zeros_like(original_images).detach().to(original_images.device)
            n_rap.requires_grad_(True)

            # 内部优化：寻找反向对抗扰动（最大化损失）
            for _ in range(10):  # 内部迭代次数可根据需要调整
                with torch.enable_grad():
                    outputs = model(perturbed_images + n_rap)
                    loss = criterion(outputs, labels)
                loss.backward()
                n_rap_grad = n_rap.grad.data
                n_rap = n_rap + (epsilon_n / 10) * n_rap_grad.sign()
                n_rap = torch.clamp(n_rap, -epsilon_n, epsilon_n)
                n_rap = n_rap.detach().requires_grad_(True)

            # 计算对抗样本的梯度（基于添加反向扰动后的损失）
            with torch.enable_grad():
                outputs = model(perturbed_images + n_rap)
                loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = perturbed_images.grad.data

            # 更新对抗样本
            perturbed_images = perturbed_images - (epsilon / num_iterations) * data_grad.sign()
            perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
            perturbed_images = perturbed_images.detach().requires_grad_(True)
        else:
            # 在晚启动之前，仅进行正常的梯度下降更新（类似于基础攻击方法）
            outputs = model(perturbed_images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = perturbed_images.grad.data
            perturbed_images = perturbed_images - (epsilon / num_iterations) * data_grad.sign()
            perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
            perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images