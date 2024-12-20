import torch
import torch.nn as nn

def BMI(model, criterion, original_images, labels, epsilon, alpha=0.001, num_iterations=10):
    perturbed_images = original_images.clone().detach().requires_grad_(True)

    for _ in range(num_iterations):
        # 计算损失
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)

        # 计算梯度
        loss.backward()

        # 更新对抗样本
        perturbation = alpha * perturbed_images.grad.sign()
        perturbed_images = perturbed_images + perturbation
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images