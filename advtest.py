import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import *
from attacks import *
import ssl
import os
from PIL import Image
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

# 定义数据预处理操作
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])

# 加载CIFAR10测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# 定义设备（GPU优先，若可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()

# 加载模型权重
weights_path = "weights/epoch_10.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))


if __name__ == "__main__":
    # 在测试集上进行FGSM攻击并评估准确率
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    epsilon = 16 / 255  # 可以调整扰动强度
    for data in testloader:
        original_images, labels = data[0].to(device), data[1].to(device)
        original_images.requires_grad = True
        
        attack_name = 'VNI-FGSM'
        if attack_name == 'FGSM':
            perturbed_images = FGSM(model, criterion, original_images, labels, epsilon)
        elif attack_name == 'BIM':
            perturbed_images = BIM(model, criterion, original_images, labels, epsilon)
        elif attack_name == 'MI-FGSM':
            perturbed_images = MI_FGSM(model, criterion, original_images, labels, epsilon)
        elif attack_name == 'NI-FGSM':
            perturbed_images = NI_FGSM(model, criterion, original_images, labels, epsilon)
        elif attack_name == 'PI-FGSM':
            perturbed_images = PI_FGSM(model, criterion, original_images, labels, epsilon)
        elif attack_name == 'VMI-FGSM':
            perturbed_images = VMI_FGSM(model, criterion, original_images, labels, epsilon)
        elif attack_name == 'VNI-FGSM':
            perturbed_images = VNI_FGSM(model, criterion, original_images, labels, epsilon)
        
        perturbed_outputs = model(perturbed_images)
        _, predicted = torch.max(perturbed_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    # Attack Success Rate
    ASR = 100 - accuracy
    print(f'Load ResNet Model Weight from {weights_path}')
    print(f'epsilon: {epsilon:.4f}')
    print(f'ASR of {attack_name} : {ASR :.2f}%')