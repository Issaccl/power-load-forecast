import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import swanlab  # ← 导入 SwanLab
import joblib  # ← 导入 joblib 用于保存 scaler

# ========== 数据加载 ==========
train = pd.read_excel(
    r'C:\Users\Lawson\Desktop\Machine Learning\深度学习\电力负荷预测\数据集\archive\train_dataframes.xlsx')
test = pd.read_excel(
    r'C:\Users\Lawson\Desktop\Machine Learning\深度学习\电力负荷预测\数据集\archive\test_dataframes.xlsx')

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])
train.set_index('datetime', inplace=True)
test.set_index('datetime', inplace=True)

feature_cols = ['week_X-2', 'week_X-3', 'week_X-4', 'MA_X-4', 'dayOfWeek', 'weekend', 'holiday',
                'Holiday_ID', 'hourOfDay', 'T2M_toc']
target_col = 'DEMAND'

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
train_X = scaler_X.fit_transform(train[feature_cols])
train_y = scaler_y.fit_transform(train[[target_col]])
test_X = scaler_X.transform(test[feature_cols])
test_y = scaler_y.transform(test[[target_col]])


def create_sequences(X, y, lookback, horizon=1):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback - horizon + 1):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback:i + lookback + horizon].flatten())
    return np.array(X_seq), np.array(y_seq)


lookback = 24
horizon = 1

X_train_seq, y_train_seq = create_sequences(train_X, train_y, lookback, horizon)
X_test_seq, y_test_seq = create_sequences(test_X, test_y, lookback, horizon)

X_train_tensor = torch.FloatTensor(X_train_seq)
y_train_tensor = torch.FloatTensor(y_train_seq)
X_test_tensor = torch.FloatTensor(X_test_seq)
y_test_tensor = torch.FloatTensor(y_test_seq)


# ========== 模型定义 ==========
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output


# ========== 超参数 ==========
input_size = 10
hidden_size = 64
num_layers = 1
output_size = 1
dropout = 0.2
batch_size = 32
learning_rate = 0.001
num_epochs = 50

# ========== 初始化 SwanLab（这里！）==========
swanlab.init(
    project="power-load-forecast",
    experiment_name="LSTM-24h-lookback",
    config={
        "model": "LSTM",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "lookback": lookback,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "feature_cols": feature_cols,
    }
)

# ========== 模型和数据加载器 ==========
model = LSTMPredictor(input_size, hidden_size, num_layers, output_size, dropout)
print(model)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ========== 训练循环 ==========
print("开始训练...")
for epoch in range(num_epochs):
    # 训练
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)

    # 验证
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(test_loader)

    # ← 记录到 SwanLab
    swanlab.log({
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }, step=epoch)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

print("训练完成！")

# ========== 测试集评估 ==========
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

y_test_np = y_test_tensor.numpy()
y_pred_np = y_pred_tensor.numpy()

y_test_actual = scaler_y.inverse_transform(y_test_np)
y_pred_actual = scaler_y.inverse_transform(y_pred_np)

mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
print(f"\n测试集 MAPE: {mape:.2f}%")

# ← 记录最终指标
swanlab.log({"test_MAPE": mape})

# ========== 保存模型和 scaler ==========
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

joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# ← 上传模型到 SwanLab
swanlab.save('lstm_model.pth')

# ← 结束 SwanLab 记录
swanlab.finish()

print("✅ 模型已保存，SwanLab 记录完成！")

# ========== 画图（可选） ==========
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(y_test_actual[:100], label='Actual')
plt.plot(y_pred_actual[:100], label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Load Demand')
plt.legend()
plt.title(f'LSTM Forecast (MAPE={mape:.2f}%)')
plt.tight_layout()
plt.show()