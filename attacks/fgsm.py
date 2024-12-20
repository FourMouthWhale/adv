import torch
import torch.nn as nn


def FGSM(model, criterion, original_images, labels, epsilon):
    outputs = model(original_images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = original_images.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = original_images + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image