import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.datasets import DatasetFolder
from PIL import Image


def loader(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 加载预训练的VGG-16模型
model = models.vgg16(pretrained=True)

# 移除VGG16的3层全连接层
classifier = list(model.classifier.children())[:-3]


num_features = model.classifier[0].in_features
classifier.extend([
    nn.Conv2d(in_channels=num_features, out_channels=512, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 2),
    nn.Softmax(dim=1)
])


new_classifier = nn.Sequential(*classifier)


model.classifier = new_classifier
model = models.vgg16(pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


for i, param in enumerate(model.features.parameters()):
    if i < 14:
        param.requires_grad = False


results = []


kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(range(10))):
    print(f"\n第 {fold + 1} 折:")


    data_folder = "data/10jwavedata/"
    train_folder = os.path.join(data_folder, f"Fold_{fold + 1}/train_data")
    test_folder = os.path.join(data_folder, f"Fold_{fold + 1}/test_data")


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 使用DatasetFolder并指定loader
    train_dataset = DatasetFolder(train_folder, loader=loader, extensions='.png', transform=transform)
    test_dataset = DatasetFolder(test_folder, loader=loader, extensions='.png', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 训练模型
    for epoch in range(3):  # 根据需要调整epoch的数量
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 19:  # Print every 20 batches
                print(f"[Fold {fold + 1}, Epoch {epoch + 1}, Batch {batch_idx + 1}] Loss: {running_loss / 20:.4f}")


    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算混淆矩阵
        confusion_mat = confusion_matrix(all_labels, all_predictions)
        print(confusion_mat)



    accuracy = correct / total
    sensitivity = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
    specificity = confusion_mat[1, 1] / (confusion_mat[1, 1] + confusion_mat[0, 1])

    results.append({
        'fold': fold + 1,
        'epoch': epoch + 1,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': confusion_mat.tolist()
    })

    print(f"准确率: {accuracy:.6f}")
    print(f"灵敏度: {sensitivity:.6f}")
    print(f"特异性: {specificity:.6f}")


avg_accuracy = np.mean([result['accuracy'] for result in results])
avg_sensitivity = np.mean([result['sensitivity'] for result in results])
avg_specificity = np.mean([result['specificity'] for result in results])


print(f"平均准确率: {avg_accuracy:.6f}")
print(f"平均灵敏度: {avg_sensitivity:.6f}")
print(f"平均特异性: {avg_specificity:.6f}")


with open('average_results.txt', 'w') as file:
    file.write(f"平均准确率: {avg_accuracy:.6f}\n")
    file.write(f"平均灵敏度: {avg_sensitivity:.6f}\n")
    file.write(f"平均特异性: {avg_specificity:.6f}\n")


with open('results.txt', 'w') as file:
    for result in results:
        file.write(f"\n第 {result['fold']} 折，Epoch {result['epoch']}：\n")
        file.write(f"准确率: {result['accuracy']:.6f}\n")
        file.write(f"灵敏度: {result['sensitivity']:.6f}\n")
        file.write(f"特异性: {result['specificity']:.6f}\n")
        file.write(f"混淆矩阵:\n{result['confusion_matrix']}\n")

# Plotting the graph
plt.figure(figsize=(10, 6))

# Set English fonts to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = True  # Resetting unicode minus

plt.plot(range(1, 11), [result['accuracy'] for result in results], label='Accuracy')
plt.plot(range(1, 11), [result['sensitivity'] for result in results], label='Sensitivity')
plt.plot(range(1, 11), [result['specificity'] for result in results], label='Specificity')

plt.xlabel('Number of folds', fontsize=14)
plt.ylabel('Performance measures (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend()

# Save the image with 600 dpi resolution
plt.savefig('accuracy_sensitivity_specificity_plot.png', dpi=600)

# Show the plot
plt.show()