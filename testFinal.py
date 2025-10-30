import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd  # 用于美化结果输出
import torch.nn.functional as F


# --- 1. 模型定义 ---
print(torch.cuda.is_available())
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. 配置和数据加载 ---

# 注意：根据您的文件结构，TEST_DATA_ROOT 应该指向 test 文件夹
TEST_DATA_ROOT = 'data/workspace/test'

# 假设模型和量化模型的文件名
ORIGINAL_MODEL_PATH = 'FruitNet_model.pth'
QUANTIZED_MODEL_PATH = 'FruitNet_model_quantized.pth'

# 类别信息 (应与训练脚本中的合并结果一致)
CLASS_NAMES = ['Apple', 'Banana', 'Guava', 'Lemon', 'Orange', 'Pomegranate']
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 64
BATCH_SIZE = 32
# 原始模型在 GPU/CPU，量化模型必须在 CPU
DEVICE_ORIGINAL = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_QUANTIZED = torch.device("cpu")  # 量化模型只能在 CPU 上运行

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载测试数据集 (使用标准的 ImageFolder，因为它已经按合并后的类别组织)
try:
    test_dataset = datasets.ImageFolder(
        root=TEST_DATA_ROOT,
        transform=data_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"成功加载测试数据。总样本数: {len(test_dataset)}")
except Exception as e:
    print(f"❌ 无法加载测试数据。请检查路径 {TEST_DATA_ROOT} 和文件夹结构。错误: {e}")
    exit()

# 检查测试集类别是否与预期匹配
if sorted(test_dataset.classes) != sorted(['Apple', 'Banana', 'Guava', 'Lemon', 'Orange', 'Pomegranate']):
    print("⚠️ 警告：测试集类别与预期不完全匹配。请检查 Lemon 映射。")
    print(f"测试集检测到的类别: {test_dataset.classes}")
    # 强制使用训练脚本中的 CLASS_NAMES，以确保混淆矩阵维度正确
    print(f"将使用训练脚本中的类别名称: {CLASS_NAMES}")


# --- 3. 模型加载函数 ---

def load_model(path, is_quantized=False, device='cpu'):
    model = SimpleCNN(num_classes=NUM_CLASSES)

    if is_quantized:
        # 1. 动态量化 (与训练脚本中保存的方式匹配)
        model = model.to('cpu')
        model_q = torch.quantization.quantize_dynamic(
            model,
            {nn.Conv2d, nn.Linear},
            dtype=torch.qint8
        )
        # 2. 加载量化模型的 state_dict
        model_q.load_state_dict(torch.load(path))
        model_q.eval()
        return model_q.to(device)
    else:
        # 原始模型加载
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model.to(device)


# --- 4. 核心测试函数 ---

def test_model(model, loader, device, model_name):
    print(f"\n--- 开始测试模型: {model_name} (在 {device}) ---")

    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0.0

    # 预热 GPU (如果使用)
    if device.type == 'cuda':
        dummy_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        _ = model(dummy_input)
        torch.cuda.synchronize()

    # 正式推理
    with torch.no_grad():
        start_time = time.time()
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.cpu().numpy()

            # 推理
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels)

        if device.type == 'cuda':
            torch.cuda.synchronize()  # 确保所有 GPU 操作完成

        end_time = time.time()
        total_time = end_time - start_time

    # --- 性能指标计算 ---

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 查准率、召回率、F1-score (使用 target_names=CLASS_NAMES)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    # 准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 推理时间 (秒/样本)
    time_per_sample = total_time / len(all_labels)

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'cm': cm,
        'report': report,
        'total_time': total_time,
        'time_per_sample': time_per_sample,
    }


# --- 5. 执行测试和对比 ---

# 5.1. 加载和测试原始模型
try:
    original_model = load_model(ORIGINAL_MODEL_PATH, is_quantized=False, device=DEVICE_ORIGINAL)
    original_results = test_model(original_model, test_loader, DEVICE_ORIGINAL, "原始模型 (Full Precision)")
except Exception as e:
    print(f"\n❌ 原始模型测试失败。请确保文件 {ORIGINAL_MODEL_PATH} 存在。错误: {e}")
    original_results = None

# 5.2. 加载和测试量化模型
try:
    quantized_model = load_model(QUANTIZED_MODEL_PATH, is_quantized=True, device=DEVICE_QUANTIZED)
    quantized_results = test_model(quantized_model, test_loader, DEVICE_QUANTIZED, "量化模型 (Quantized)")
except Exception as e:
    print(f"\n❌ 量化模型测试失败。请确保文件 {QUANTIZED_MODEL_PATH} 存在。错误: {e}")
    quantized_results = None

# --- 6. 结果对比输出 ---

print("\n" + "=" * 80)
print("             🚀 模型性能对比报告 (测试集) 🚀")
print("=" * 80 + "\n")

results = [original_results, quantized_results]
results = [r for r in results if r is not None]

if not results:
    print("没有可用的测试结果进行对比。")
    exit()

# 6.1. 混淆矩阵对比
for res in results:
    print(f"### {res['model_name']} - 混淆矩阵 (CM) ###")
    cm_df = pd.DataFrame(res['cm'], index=CLASS_NAMES, columns=CLASS_NAMES)
    print("--- 预测标签 (列) ---")
    print(cm_df)
    print("--- 真实标签 (行) ---\n")

# 6.2. 汇总指标表格

summary_data = []
for res in results:

    macro_avg = res['report']['macro avg']
    weighted_avg = res['report']['weighted avg']

    # 重点指标提取
    summary_data.append({
        '模型': res['model_name'],
        # 使用已经计算好的整体准确率
        '准确率 (Accuracy)': f"{res['accuracy']:.4f}",
        '查准率 (P_wtd)': f"{weighted_avg['precision']:.4f}",
        '召回率 (R_wtd)': f"{weighted_avg['recall']:.4f}",
        'F1-Score (wtd)': f"{weighted_avg['f1-score']:.4f}",
        '总推理时间 (s)': f"{res['total_time']:.4f}",
        '平均每样本时间 (ms)': f"{res['time_per_sample'] * 1000:.4f}",
    })

summary_df = pd.DataFrame(summary_data)
summary_df.set_index('模型', inplace=True)

print("### 📚 关键指标汇总对比 ###")
print(summary_df)

print("\n" + "=" * 80)

# 6.3. 分类报告 (可选，提供更详细的每类别指标)
for res in results:
    print(f"\n### {res['model_name']} - 详细分类报告 ###")
    report_df = pd.DataFrame(res['report']).transpose()
    # 格式化输出
    print(report_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (float, np.float64)) else x))

print("=" * 80)