import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

train=pd.read_excel(r'C:\Users\Lawson\Desktop\Machine Learning\深度学习\电力负荷预测\数据集\archive\train_dataframes.xlsx')
test=pd.read_excel(r'C:\Users\Lawson\Desktop\Machine Learning\深度学习\电力负荷预测\数据集\archive\test_dataframes.xlsx')


#print("\n缺失值统计：\n",train.isnull().sum())
train['datetime']=pd.to_datetime(train['datetime'])
test['datetime']=pd.to_datetime(test['datetime'])
train.set_index('datetime',inplace=True)
test.set_index('datetime',inplace=True)


#画负荷曲线
# plt.figure(figsize=(12,4))
# plt.plot(train.index[:1000],train['DEMAND'][:1000])
# plt.title('Load Deamnd')
# plt.show()

feature_cols = ['week_X-2', 'week_X-3', 'week_X-4', 'MA_X-4', 'dayOfWeek', 'weekend', 'holiday',
                'Holiday_ID','hourOfDay','T2M_toc' ]
target_col='DEMAND'

scaler_X=MinMaxScaler()
scaler_y=MinMaxScaler()
#拟合训练集并转换
train_X=scaler_X.fit_transform(train[feature_cols])
train_y=scaler_y.fit_transform(train[[target_col]])
#转换测试集
test_X=scaler_X.transform(test[feature_cols])
test_y=scaler_y.transform(test[[target_col]])

# print("训练集特征形状:", train_X.shape)
# print("训练集目标形状:", train_y.shape)
# print("测试集特征形状:", test_X.shape)
# print("测试集目标形状:", test_y.shape)
#
# print("特征列数:", len(feature_cols))
# print("train_X 列数:", train_X.shape[1])
# print("特征列名:", feature_cols)


def create_sequences(X, y, lookback, horizon=1):
    """
    X: numpy array (n_samples, n_features)
    y: numpy array (n_samples, 1)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback - horizon + 1):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback:i+lookback+horizon].flatten())  # flatten 是 numpy 的
    return np.array(X_seq), np.array(y_seq)

# 参数
lookback = 24
horizon = 1

# 生成序列
X_train_seq, y_train_seq = create_sequences(train_X, train_y, lookback, horizon)
X_test_seq, y_test_seq = create_sequences(test_X, test_y, lookback, horizon)

# 转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train_seq)
y_train_tensor = torch.FloatTensor(y_train_seq)
X_test_tensor = torch.FloatTensor(X_test_seq)
y_test_tensor = torch.FloatTensor(y_test_seq)

# print("训练集输入形状:", X_train_tensor.shape)  # (36696, 24, 10)
# print("训练集输出形状:", y_train_tensor.shape)  # (36696, 1)
# print("测试集输入形状:", X_test_tensor.shape)   # (144, 24, 10)
# print("测试集输出形状:", y_test_tensor.shape)   # (144, 1)

class LSTMPredictor(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,dropout=0.2):
        super(LSTMPredictor,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.lstm=nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers >1 else 0
        )

        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        lstm_out,(h_n,c_n)=self.lstm(x)
        last_output=lstm_out[:,-1,:]
        last_output=self.dropout(last_output)
        output=self.fc(last_output)
        return output
input_size = 10          # 特征数
hidden_size = 64         # LSTM 隐藏单元数
num_layers = 1           # LSTM 层数
output_size = 1          # 预测下一小时
dropout = 0.2

# 创建模型
model = LSTMPredictor(input_size, hidden_size, num_layers, output_size, dropout)

# 打印模型结构
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

#训练开始！


import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 设置参数
batch_size = 32
learning_rate = 0.001
num_epochs = 50

# 2. 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. 训练循环
train_losses = []
val_losses = []

print("开始训练...")
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    epoch_train_loss = 0

    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证模式（用测试集）
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    # 每10个epoch打印一次
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

print("训练完成！")

# 5. 画损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# 6. 测试集预测
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

# 转换为 numpy
y_test_np = y_test_tensor.numpy()
y_pred_np = y_pred_tensor.numpy()

# 7. 反归一化
y_test_actual = scaler_y.inverse_transform(y_test_np)
y_pred_actual = scaler_y.inverse_transform(y_pred_np)

# 8. 计算 MAPE
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
print(f"\n测试集 MAPE: {mape:.2f}%")

# 9. 画预测对比图
plt.subplot(1, 2, 2)
plt.plot(y_test_actual[:100], label='Actual', linewidth=1.5)
plt.plot(y_pred_actual[:100], label='Predicted', linewidth=1.5)
plt.xlabel('Time Step')
plt.ylabel('Load Demand')
plt.legend()
plt.title(f'LSTM Forecast vs Actual (MAPE={mape:.2f}%)')
plt.tight_layout()
plt.show()

# ========== 在你的训练代码最后添加 ==========

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'output_size': output_size,
    'dropout': dropout,
    'lookback': lookback,
    'feature_cols': feature_cols,
}, 'lstm_model.pth')

# 保存归一化器
import joblib
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

print("模型和归一化器已保存！")










