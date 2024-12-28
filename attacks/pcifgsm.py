import torch
import torch.nn as nn

def PC_I_FGSM(model, criterion, original_images, labels, epsilon, num_iterations=10, num_predictions=1):
    """
    PC-I-FGSM (Prediction-Correction Iterative Fast Gradient Sign Method)

    参数:
    - model: 要攻击的模型
    - criterion: 损失函数
    - original_images: 原始图像
    - labels: 原始图像的标签
    - epsilon: 最大扰动幅度
    - num_iterations: 迭代次数
    - num_predictions: 预测次数
    """
    alpha = epsilon / num_iterations
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    ori_perturbed_images = original_images.clone().detach().requires_grad_(True)

    # 用于校正阶段
    original_outputs = model(ori_perturbed_images)
    original_loss = criterion(original_outputs, labels)
    model.zero_grad()
    original_loss.backward()
    original_gradient = ori_perturbed_images.grad.data

    for _ in range(num_iterations):
        # 预测阶段
        acumulated_predicted_gradients = torch.zeros_like(original_images).detach().to(original_images.device)
        # 先更新一次对抗样本
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_images.grad.data
        perturbed_images = perturbed_images.detach().requires_grad_(True)
        for _ in range(num_predictions-1):
            outputs = model(perturbed_images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = perturbed_images.grad.data
            # 更新对抗样本（预测步骤）
            perturbed_images = perturbed_images + alpha * data_grad.sign()
            perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
            perturbed_images = perturbed_images.detach().requires_grad_(True)
            acumulated_predicted_gradients += data_grad / torch.sum(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)

        # 校正阶段
        corrected_gradient = original_gradient + acumulated_predicted_gradients
        # 更新对抗样本（校正步骤）
        perturbed_images = original_images + epsilon * corrected_gradient.sign()
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images