import torch
import torch.nn as nn

def VA_I_FGSM(model, criterion, original_images, labels, epsilon, num_iterations=10, num_aux_labels=3):
    """
    VA-I-FGSM (Virtual Step and Auxiliary Gradients Iterative Fast Gradient Sign Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - virtual_step_size: 虚拟步长
    - num_aux_labels: 辅助标签数量
    """
    # 虚拟步长
    alpha = epsilon / num_iterations    
    # 复制原始图像作为初始的对抗样本
    ori_perturbed_images = original_images.clone().detach().requires_grad_(True)
    perturbed_images = original_images.clone().detach().requires_grad_(True)

    for _ in range(num_iterations):
        outputs = model(ori_perturbed_images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        # 计算基于真实标签的梯度
        main_grad = ori_perturbed_images.grad.data.sign()
        perturbed_images = perturbed_images + alpha * main_grad

        # 迭代num_aux_labels次数
        for _ in range(num_aux_labels):
            # 每次迭代中，都随机生成张量aux_labels，其与labels尺寸相同
            aux_labels = torch.randint(low=0, high=10, size=labels.size(), device=original_images.device)

            # 检查并替换与真实标签相同的辅助标签
            mask = aux_labels == labels
            while mask.any():
                aux_labels[mask] = torch.randint(low=0, high=10, size=(mask.sum(),), device=original_images.device)
                mask = aux_labels == labels

            # 计算辅助标签的损失
            outputs = model(ori_perturbed_images)
            aux_loss = criterion(outputs, aux_labels)
            model.zero_grad()
            aux_loss.backward()
            aux_grad = ori_perturbed_images.grad.data.sign()
            perturbed_images = perturbed_images - alpha * aux_grad
            ori_perturbed_images = ori_perturbed_images.detach().requires_grad_(True)

        ori_perturbed_images = ori_perturbed_images.detach().requires_grad_(True)

    # 确保对抗样本在epsilon范围内
    perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)

    return perturbed_images