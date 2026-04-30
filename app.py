import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="电力负荷预测", page_icon="⚡", layout="wide")

# ========== LSTM 模型定义 ==========
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)

# ========== 加载模型 ==========
@st.cache_resource
def load_model():
    checkpoint = torch.load('lstm_model.pth', map_location='cpu', weights_only=False)
    model = LSTMPredictor(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['num_layers'], checkpoint['output_size'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y, checkpoint['feature_cols'], checkpoint['lookback']

# ========== 加载示例数据 ==========
@st.cache_data
def load_demo_data():
    train = pd.read_excel('data/train_dataframes.xlsx', engine='openpyxl')
    test = pd.read_excel('data/test_dataframes.xlsx', engine='openpyxl')
    train['datetime'] = pd.to_datetime(train['datetime'])
    test['datetime'] = pd.to_datetime(test['datetime'])
    train.set_index('datetime', inplace=True)
    test.set_index('datetime', inplace=True)
    return train, test

# ========== 预测函数 ==========
def predict(model, scaler_X, scaler_y, last_sequence, feature_cols, steps=24):
    predictions = []
    seq = last_sequence.copy()
    for _ in range(steps):
        X = scaler_X.transform(seq[feature_cols].values[-24:])
        X_tensor = torch.FloatTensor(X).unsqueeze(0)
        with torch.no_grad():
            pred = model(X_tensor).numpy()[0, 0]
        pred_actual = scaler_y.inverse_transform([[pred]])[0, 0]
        predictions.append(pred_actual)
        new_row = seq.iloc[-1:].copy()
        new_row['DEMAND'] = pred_actual
        new_row.index = [seq.index[-1] + timedelta(hours=1)]
        seq = pd.concat([seq.iloc[1:], new_row])
    return np.array(predictions)

# ========== 主界面 ==========
st.title("⚡ 基于LSTM的电力负荷预测系统")

# 加载模型
model, scaler_X, scaler_y, feature_cols, lookback = load_model()
st.sidebar.success("✅ 模型加载成功")

# 数据来源选择
st.sidebar.subheader("📁 数据来源")
data_source = st.sidebar.radio("选择数据来源", ["📦 使用示例数据", "📤 上传我的数据"])

# 预测参数
predict_hours = st.sidebar.slider("预测时长（小时）", 1, 72, 24)

# ========== 获取数据 ==========
if data_source == "📦 使用示例数据":
    train, test = load_demo_data()
    # 选择日期
    dates = sorted(test.index.date)
    selected_date = st.sidebar.date_input("选择日期", dates[0], min_value=dates[0], max_value=dates[-1])
    
    # 取数据
    start_idx = test.index.get_loc(pd.Timestamp(selected_date).strftime('%Y-%m-%d'))
    start_idx = max(start_idx, lookback)
    historical = test.iloc[max(0, start_idx-48):start_idx]
    last_seq = test.iloc[start_idx-lookback:start_idx]

else:
    uploaded_file = st.sidebar.file_uploader("上传CSV文件（需包含过去24小时数据）", type=['csv'])
    if uploaded_file:
        user_data = pd.read_csv(uploaded_file)
        user_data['datetime'] = pd.to_datetime(user_data['datetime'])
        user_data.set_index('datetime', inplace=True)
        
        # 检查必要列
        missing = set(feature_cols + ['DEMAND']) - set(user_data.columns)
        if missing:
            st.error(f"缺少列: {missing}")
            st.stop()
        
        if len(user_data) < lookback:
            st.error(f"需要至少{lookback}小时数据")
            st.stop()
        
        historical = user_data.iloc[-48:] if len(user_data) >= 48 else user_data
        last_seq = user_data.iloc[-lookback:]
    else:
        st.info("👆 请在侧边栏上传CSV文件，或切换到示例数据")
        st.stop()

# ========== 预测 ==========
with st.spinner("🔄 预测中..."):
    predictions = predict(model, scaler_X, scaler_y, last_seq, feature_cols, predict_hours)

# ========== 指标 ==========
col1, col2, col3 = st.columns(3)
col1.metric("当前负荷", f"{historical['DEMAND'].iloc[-1]:.1f} MW")
col2.metric("预测峰值", f"{predictions.max():.1f} MW")
col3.metric("预测谷值", f"{predictions.min():.1f} MW")

# ========== 图表 ==========
st.subheader("📊 负荷预测曲线")

last_time = historical.index[-1]
hist_times = historical.index
fut_times = [last_time + timedelta(hours=i+1) for i in range(predict_hours)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_times, y=historical['DEMAND'], name='历史', line=dict(color='#1f77b4', width=2)))
fig.add_trace(go.Scatter(x=fut_times, y=predictions, name='预测', line=dict(color='#ff7f0e', width=3, dash='dash')))
fig.update_layout(height=500, hovermode='x unified')
fig.update_xaxes(title="时间")
fig.update_yaxes(title="负荷 (MW)")

st.plotly_chart(fig, width='stretch')

# ========== 表格 ==========
st.subheader("📋 预测数据")
df = pd.DataFrame({'时间': [t.strftime('%Y-%m-%d %H:00') for t in fut_times], '预测负荷(MW)': predictions.round(2)})
st.dataframe(df, width='stretch', hide_index=True)

# 下载
csv = df.to_csv(index=False).encode('utf-8-sig')
st.download_button("📥 下载预测结果", csv, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 'text/csv')
