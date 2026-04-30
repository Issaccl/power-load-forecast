import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta

st.set_page_config(page_title="电力负荷预测", page_icon="⚡", layout="wide")

# ========== LSTM 模型 ==========
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))

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
    for df in [train, test]:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    return train, test

# ========== 预测 ==========
def predict(model, scaler_X, scaler_y, seq, feature_cols, steps=24):
    preds = []
    s = seq.copy()
    for _ in range(steps):
        X = scaler_X.transform(s[feature_cols].values[-24:])
        with torch.no_grad():
            p = model(torch.FloatTensor(X).unsqueeze(0)).numpy()[0, 0]
        p_act = scaler_y.inverse_transform([[p]])[0, 0]
        preds.append(p_act)
        new = s.iloc[-1:].copy()
        new['DEMAND'] = p_act
        new.index = [s.index[-1] + timedelta(hours=1)]
        s = pd.concat([s.iloc[1:], new])
    return np.array(preds)

# ========== 主界面 ==========
st.title("⚡ 基于LSTM的电力负荷预测系统")

model, scaler_X, scaler_y, feature_cols, lookback = load_model()
st.sidebar.success("✅ 模型加载成功")

st.sidebar.subheader("📁 数据来源")
data_source = st.sidebar.radio("选择", ["📦 示例数据", "📤 上传CSV"])
predict_hours = st.sidebar.slider("预测时长（小时）", 1, 72, 24)

if data_source == "📦 示例数据":
    train, test = load_demo_data()
    dates = sorted(set(test.index.date))
    d = st.sidebar.date_input("选择日期", dates[0], min_value=dates[0], max_value=dates[-1])
    mask = test.index.date == d
    idx = np.where(mask)[0]
    if len(idx) == 0:
        st.error("无数据")
        st.stop()
    start = idx[0]
    if start < lookback:
        start = lookback
    historical = test.iloc[max(0, start-48):start]
    last_seq = test.iloc[start-lookback:start]
else:
    f = st.sidebar.file_uploader("上传CSV", type=['csv'])
    if f is None:
        st.info("👆 上传CSV文件或切换到示例数据")
        st.stop()
    ud = pd.read_csv(f)
    ud['datetime'] = pd.to_datetime(ud['datetime'])
    ud.set_index('datetime', inplace=True)
    missing = set(feature_cols + ['DEMAND']) - set(ud.columns)
    if missing:
        st.error(f"缺少列: {missing}")
        st.stop()
    if len(ud) < lookback:
        st.error(f"至少需要{lookback}小时数据")
        st.stop()
    historical = ud.iloc[-48:] if len(ud) >= 48 else ud
    last_seq = ud.iloc[-lookback:]

predictions = predict(model, scaler_X, scaler_y, last_seq, feature_cols, predict_hours)

c1, c2, c3 = st.columns(3)
c1.metric("当前负荷", f"{historical['DEMAND'].iloc[-1]:.1f} MW")
c2.metric("预测峰值", f"{predictions.max():.1f} MW")
c3.metric("预测谷值", f"{predictions.min():.1f} MW")

st.subheader("📊 负荷预测曲线")
lt = historical.index[-1]
ft = [lt + timedelta(hours=i+1) for i in range(predict_hours)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=historical.index, y=historical['DEMAND'], name='历史', line=dict(color='#1f77b4', width=2)))
fig.add_trace(go.Scatter(x=ft, y=predictions, name='预测', line=dict(color='#ff7f0e', width=3, dash='dash')))
fig.update_layout(height=450, hovermode='x unified')
st.plotly_chart(fig, width='stretch')

st.subheader("📋 预测数据")
df = pd.DataFrame({'时间': [t.strftime('%Y-%m-%d %H:00') for t in ft], '预测负荷(MW)': predictions.round(2)})
st.dataframe(df, width='stretch', hide_index=True)

st.download_button("📥 下载结果", df.to_csv(index=False).encode('utf-8-sig'), f"pred_{datetime.now():%Y%m%d_%H%M}.csv", 'text/csv')
