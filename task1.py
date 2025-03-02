import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet("hf://datasets/dnagpt/dna_core_promoter/data/train-00000-of-00001.parquet")

# 对数据集进行预处理（One-Hot编码，将DNA序列中的每个碱基映射为四维向量）
def one_hot_encode(sequence):
    encoding = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([encoding[base] for base in sequence])

sequences = df['sequence'].apply(one_hot_encode).values
X = np.array([seq for seq in sequences])
y = df['label'].values

# 划分训练集和测试集，并将数据转换为PyTorch张量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 构建CNN/LSTM混合模型
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN_LSTM_Model, self).__init__()
        # CNN层
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        # LSTM层
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x): # 定义前向传播过程
        # CNN层
        x = x.permute(0, 2, 1)  # 交换维度
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # LSTM层
        x = x.permute(0, 2, 1)
        lstm_out, (hn, cn) = self.lstm(x)
        x = hn[-1] # 只取LSTM输出的最后一个时刻的隐藏状态
        # 全连接层
        x = self.fc(x)
        return x

# 训练模型
model = CNN_LSTM_Model(input_size=128, hidden_size=64, output_size=2)

# 使用Adam优化器和交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad(): # 禁用梯度计算
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")
