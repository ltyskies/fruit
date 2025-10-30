import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import numpy as np
from tqdm import tqdm
import joblib

# --- 0. 类别映射和自定义 Dataset ---

# 18 个旧类别 -> 6 个新类别 (类别名称需要根据实际文件夹名称确定)
# 假设您的原始文件夹名就是输出的类别名称：
OLD_CLASSES = ['Apple', 'Apple_Bad', 'Apple_Good', 'Banana', 'Banana_Bad', 'Banana_Good',
               'Guava', 'Guava_Bad', 'Guava_Good', 'Lemon', 'Lemon_Bad', 'Lemon_Good',
               'Orange', 'Orange_Bad', 'Orange_Good', 'Pomegranate', 'Pomegranate_Bad', 'Pomegranate_Good']

# 定义新的类别名称
NEW_CLASS_NAMES = ['Apple', 'Banana', 'Guava', 'Lemon', 'Orange', 'Pomegranate']

# 定义映射规则
class_mapping = {}
for old_cls in OLD_CLASSES:
    if old_cls.startswith('Apple'):
        class_mapping[old_cls] = 'Apple'
    elif old_cls.startswith('Banana'):
        class_mapping[old_cls] = 'Banana'
    elif old_cls.startswith('Guava'):
        class_mapping[old_cls] = 'Guava'
    # 注意：'Lemon' 和 'Lime' 合并为一个类别，我们命名为 'Lime/Lemon'
    elif old_cls.startswith('Lemon'):
        class_mapping[old_cls] = 'Lemon'
    elif old_cls.startswith('Orange'):
        class_mapping[old_cls] = 'Orange'
    elif old_cls.startswith('Pomegranate'):
        class_mapping[old_cls] = 'Pomegranate'

# 创建新标签到数字索引的映射
new_class_to_idx = {name: i for i, name in enumerate(NEW_CLASS_NAMES)}


# 自定义 Dataset 类进行标签重映射
class MergedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

        self.new_class_to_idx = new_class_to_idx
        self.new_classes = NEW_CLASS_NAMES

        # 原始标签 (self.targets) 基于 ImageFolder 的默认排序，需要重新映射
        self.merged_targets = []
        for old_idx in self.targets:
            # 1. 找到原始类别名称
            old_class_name = self.classes[old_idx]
            # 2. 找到新的类别名称
            new_class_name = class_mapping.get(old_class_name)
            if new_class_name is None:
                raise ValueError(f"无法找到旧类别 {old_class_name} 的新映射。请检查 class_mapping。")
            # 3. 找到新的数字索引
            new_idx = self.new_class_to_idx[new_class_name]
            self.merged_targets.append(new_idx)

    def __getitem__(self, index):
        # 调用父类的 __getitem__ 来获取图像和原始标签
        path, _ = self.samples[index]

        # 🐛 错误修正：ImageFolder 的加载器是 self.loader (没有下划线)
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        # 返回图像和新的合并标签
        return sample, self.merged_targets[index]

    # 覆盖 len 以保持一致性
    def __len__(self):
        return len(self.samples)


# --- 1. 模型定义 ---

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, num_classes)  # 使用新的 num_classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. 配置和数据加载 ---

ROOT_DIR = 'data/workspace/FruitNet'
TRAIN_DATA_PATH = os.path.join(ROOT_DIR)

# 超参数
BATCH_SIZE = 32
IMAGE_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用自定义的 MergedImageFolder 加载数据
DATA_ROOT = os.path.join(ROOT_DIR)
dataset = MergedImageFolder(
    root=DATA_ROOT,
    transform=data_transforms
)

# 自动获取类别数量 (现在是合并后的数量)
NUM_CLASSES = len(dataset.new_classes)  # 使用 MergedImageFolder 中的新类别列表
CLASS_NAMES = dataset.new_classes
print(f"检测到的合并类别数量: {NUM_CLASSES}")
print(f"合并后的类别名称: {CLASS_NAMES}")

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. 实例化模型、损失函数和优化器 ---

# 实例化模型 (使用新的 NUM_CLASSES=6)
model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- 4. 评估函数 (保持不变) ---

def evaluate_model(model, loader, device, name="Validation", class_names=None):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    print(f"\n--- 开始在 {name} 集上评估 ---")

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Evaluating {name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"{name} 集准确率: {accuracy:.2f}%")

    print("\n--- 预测结果示例 (前 10 个) ---")
    for i in range(min(10, len(all_labels))):
        true_label_idx = all_labels[i]
        pred_label_idx = all_preds[i]

        # 使用合并后的类别名称进行展示
        true_name = class_names[true_label_idx] if class_names and true_label_idx < len(class_names) else str(
            true_label_idx)
        pred_name = class_names[pred_label_idx] if class_names and pred_label_idx < len(class_names) else str(
            pred_label_idx)

        print(
            f"样本 {i + 1}: 真实标签={true_name} ({true_label_idx}), 预测标签={pred_name} ({pred_label_idx}) {'✅' if true_label_idx == pred_label_idx else '❌'}")

    return accuracy


# --- 5. 训练函数 ---

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, class_names):
    best_val_accuracy = 0.0
    best_model_path = 'FruitNet_best_model_temp.pth'

    print("\n--- 开始训练 ---")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [TRAIN]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        val_accuracy = evaluate_model(model, val_loader, device, name="Validation", class_names=class_names)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"*** 当前验证集准确率 {best_val_accuracy:.2f}% 优于历史最高，模型已保存到 {best_model_path} ***")

    print("--- 训练完成 ---")
    return best_model_path


# --- 6. 执行训练和评估 ---

best_model_path = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE, CLASS_NAMES)

# 加载最优模型用于最终评估和导出
best_model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)  # 使用新的 NUM_CLASSES 重新实例化
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()
print("\n--- 最终评估 (加载最优模型) ---")
evaluate_model(best_model, val_loader, DEVICE, name="Final Validation", class_names=CLASS_NAMES)

# --- 7. 模型导出和量化 ---

print("\n--- 开始模型导出和轻量化 ---")

# 7.1. 导出为 .pth (Final Model)
final_pth_path = 'FruitNet_model.pth'
torch.save(best_model.state_dict(), final_pth_path)
print(f"✅ 模型状态字典已保存到 {final_pth_path}")

# 7.2. 导出为 .joblib (保存 state_dict)
try:
    joblib_path = 'FruitNet_model.joblib'
    joblib.dump(best_model.state_dict(), joblib_path)
    print(f"✅ 模型状态字典已使用 joblib 保存到 {joblib_path}")
except Exception as e:
    print(f"❌ 导出 joblib 失败: {e}")

# 7.3. 导出为 .onnx
try:
    onnx_path = 'FruitNet_model.onnx'
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    torch.onnx.export(
        best_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ 模型已导出为 ONNX 格式到 {onnx_path}")
except Exception as e:
    print(f"❌ 导出 ONNX 失败: {e}")

# 7.4. 模型轻量化 (动态量化)
quantized_path = 'FruitNet_model_quantized.pth'
try:
    quantized_model = SimpleCNN(num_classes=NUM_CLASSES)
    quantized_model.load_state_dict(torch.load(best_model_path))
    quantized_model.eval()

    quantized_model_cpu = quantized_model.to('cpu')
    quantized_model_dyn = torch.quantization.quantize_dynamic(
        quantized_model_cpu,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )

    torch.save(quantized_model_dyn.state_dict(), quantized_path)
    print(f"✅ 模型已通过 **动态量化** 轻量化并保存到 {quantized_path}")

    original_size = os.path.getsize(final_pth_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"模型大小对比: 原始 ({original_size:.2f} MB) vs. 量化 ({quantized_size:.2f} MB)")

except Exception as e:
    print(f"❌ 模型轻量化 (动态量化) 失败: {e}")