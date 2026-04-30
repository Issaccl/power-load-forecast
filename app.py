# app.py - 电力负荷预测Streamlit应用
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# ========== 页面配置 ==========
st.set_page_config(
    page_title="电力负荷预测系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========== LSTM模型定义（必须和训练时一致） ==========
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


# ========== 加载模型和scaler ==========
@st.cache_resource
def load_model_and_scalers():
    """加载训练好的模型和归一化器"""
    try:
        # 加载模型参数
        checkpoint = torch.load('lstm_model.pth', map_location=torch.device('cpu'))

        # 重建模型
        model = LSTMPredictor(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            output_size=checkpoint['output_size'],
            dropout=checkpoint['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 加载归一化器
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')

        # 加载特征列名
        feature_cols = checkpoint['feature_cols']
        lookback = checkpoint['lookback']

        return model, scaler_X, scaler_y, feature_cols, lookback
    except FileNotFoundError as e:
        st.error(f"❌ 模型文件未找到: {str(e)}")
        st.info("请确保以下文件在应用目录中：\n- lstm_model.pth\n- scaler_X.pkl\n- scaler_y.pkl")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"❌ 加载模型时出错: {str(e)}")
        return None, None, None, None, None


# ========== 预测函数 ==========
def predict_future(model, scaler_X, scaler_y, last_sequence, feature_cols, steps=24):
    """
    预测未来负荷
    last_sequence: DataFrame，最后lookback小时的数据
    steps: 预测步数
    """
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(steps):
        # 准备输入
        X = scaler_X.transform(current_sequence[feature_cols])
        X_tensor = torch.FloatTensor(X).unsqueeze(0)  # (1, lookback, n_features)

        # 预测
        with torch.no_grad():
            pred = model(X_tensor).numpy()[0, 0]

        # 反归一化
        pred_reshaped = np.array([[pred]])
        pred_actual = scaler_y.inverse_transform(pred_reshaped)[0, 0]
        predictions.append(pred_actual)

        # 更新序列（简化版：复制最后一行并更新需求值）
        new_row = current_sequence.iloc[-1:].copy()
        new_row['DEMAND'] = pred_actual
        # 更新时间特征（简化处理）
        new_row.index = [current_sequence.index[-1] + timedelta(hours=1)]
        current_sequence = pd.concat([current_sequence.iloc[1:], new_row])

    return np.array(predictions)


# ========== 加载数据 ==========
@st.cache_datadef 
def load_data():
    """加载训练和测试数据"""
    try:
        import os
        train = pd.read_excel(os.path.join('data', 'train_dataframes.xlsx'), engine='openpyxl')
        test = pd.read_excel(os.path.join('data', 'test_dataframes.xlsx'), engine='openpyxl')
        
        train['datetime'] = pd.to_datetime(train['datetime'])
        test['datetime'] = pd.to_datetime(test['datetime'])
        train.set_index('datetime', inplace=True)
        test.set_index('datetime', inplace=True)
        
        return train, test
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None, None

# ========== 主应用 ==========
def main():
    st.title("⚡ 基于LSTM的电力负荷预测系统")
    st.markdown("---")

    # 加载模型
    model, scaler_X, scaler_y, feature_cols, lookback = load_model_and_scalers()

    if model is None:
        st.warning("请先训练模型并保存文件")
        return

    st.sidebar.success("✅ 模型加载成功")

    # 加载数据
    train_data, test_data = load_data()

    if train_data is None:
        st.error("❌ 数据文件未找到")
        return

    # ========== 侧边栏 ==========
    st.sidebar.title("📊 控制面板")

    # 预测设置
    st.sidebar.subheader("🔮 预测参数")
    predict_hours = st.sidebar.slider(
        "预测时长（小时）",
        min_value=1,
        max_value=72,
        value=24,
        step=1
    )

    # 数据选择
    st.sidebar.subheader("📅 数据选择")
    use_test_data = st.sidebar.checkbox("使用测试集数据", value=True)

    if use_test_data:
        available_dates = test_data.index.date
        selected_date = st.sidebar.date_input(
            "选择起始日期",
            value=available_dates[0] if len(available_dates) > 0 else datetime.now().date(),
            min_value=available_dates[0] if len(available_dates) > 0 else None,
            max_value=available_dates[-1] if len(available_dates) > 0 else None
        )

    # 高级设置
    st.sidebar.subheader("⚙️ 高级设置")
    show_confidence = st.sidebar.checkbox("显示置信区间", value=True)
    confidence_level = st.sidebar.slider("置信水平", 0.8, 0.99, 0.95, 0.01)

    # ========== 主页面 ==========
    # 准备历史数据
    if use_test_data:
        # 使用测试集中选中的日期
        try:
            start_idx = test_data.index.get_loc(pd.Timestamp(selected_date))
            if start_idx < lookback:
                st.warning(f"需要至少{lookback}小时的历史数据，已使用最早可用数据")
                start_idx = lookback

            historical_data = test_data.iloc[max(0, start_idx - 48):start_idx]  # 显示48小时历史
            last_sequence = test_data.iloc[start_idx - lookback:start_idx]
        except:
            st.error("所选日期数据不足，使用默认数据")
            historical_data = test_data.iloc[:48]
            last_sequence = test_data.iloc[:lookback]
    else:
        # 使用训练集最后的数据
        historical_data = train_data.iloc[-48:]
        last_sequence = train_data.iloc[-lookback:]

    historical_load = historical_data['DEMAND'].values

    # 执行预测
    with st.spinner("🔄 正在进行LSTM预测..."):
        predictions = predict_future(
            model, scaler_X, scaler_y,
            last_sequence, feature_cols,
            steps=predict_hours
        )

    # ========== 统计指标 ==========
    st.subheader("📈 预测结果概览")

    col1, col2, col3, col4 = st.columns(4)

    current_load = historical_load[-1]
    pred_peak = predictions.max()
    pred_valley = predictions.min()
    load_factor = (predictions.mean() / pred_peak * 100) if pred_peak > 0 else 0

    col1.metric(
        "当前负荷",
        f"{current_load:.1f} MW",
        delta=f"{predictions[0] - current_load:.1f} MW"
    )

    col2.metric(
        "预测峰值",
        f"{pred_peak:.1f} MW",
        delta=f"{pred_peak - current_load:.1f} MW"
    )

    col3.metric(
        "预测谷值",
        f"{pred_valley:.1f} MW"
    )

    col4.metric(
        "平均负荷率",
        f"{load_factor:.1f}%"
    )

    # ========== 可视化 ==========
    st.subheader("负荷预测曲线")

    # 时间轴
    last_time = historical_data.index[-1]
    historical_times = historical_data.index
    future_times = [last_time + timedelta(hours=i + 1) for i in range(predict_hours)]

    # 创建Plotly图表
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('负荷预测曲线', '预测统计'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )

    # 历史数据
    fig.add_trace(
        go.Scatter(
            x=historical_times,
            y=historical_load,
            mode='lines',
            name='历史负荷',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )

    # 预测数据
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=predictions,
            mode='lines+markers',
            name='预测负荷',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # 置信区间
    if show_confidence:
        # 简化的置信区间计算
        historical_std = historical_load[-24:].std()
        z_score = {
            0.80: 1.28, 0.85: 1.44, 0.90: 1.645,
            0.95: 1.96, 0.99: 2.576
        }
        z = z_score.get(confidence_level, 1.96)

        upper = predictions + z * historical_std
        lower = predictions - z * historical_std

        fig.add_trace(
            go.Scatter(
                x=future_times + future_times[::-1],
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name=f'{confidence_level * 100:.0f}% 置信区间'
            ),
            row=1, col=1
        )

    # 柱状图显示预测分布
    fig.add_trace(
        go.Bar(
            x=['峰值', '均值', '谷值'],
            y=[pred_peak, predictions.mean(), pred_valley],
            marker_color=['#ff4444', '#ffaa00', '#00cc44'],
            name='预测统计'
        ),
        row=2, col=1
    )

    # 布局更新
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(title_text="时间", row=1, col=1)
    fig.update_yaxes(title_text="负荷 (MW)", row=1, col=1)
    fig.update_yaxes(title_text="负荷 (MW)", row=2, col=1)

    st.plotly_chart(fig, width='stretch')

    # ========== 预测数据表格 ==========
    st.subheader("📋 详细预测数据")

    # 创建数据表
    df_predictions = pd.DataFrame({
        '时间': [t.strftime('%Y-%m-%d %H:00') for t in future_times],
        '预测负荷(MW)': predictions.round(2),
        '变化趋势': ['↑' if i > 0 and predictions[i] > predictions[i - 1]
                     else '↓' if i > 0 else '→'
                     for i in range(len(predictions))]
    })

    # 样式化显示
    st.dataframe(
        df_predictions,
        width='stretch',
        hide_index=True,
        column_config={
            "变化趋势": st.column_config.TextColumn(
                "趋势",
                help="与前一时刻相比的变化趋势",
                width="small"
            )
        }
    )

    # 下载按钮
    csv = df_predictions.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载预测数据 (CSV)",
        data=csv,
        file_name=f'load_prediction_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv'
    )

    # ========== 模型信息 ==========
    with st.expander("🔍 模型详细信息"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **模型架构：**
            - 类型：LSTM 神经网络
            - 输入特征数：{len(feature_cols)}
            - 回顾窗口：{lookback} 小时
            - 隐藏层大小：{64}
            - LSTM层数：{1}
            """)

        with col2:
            st.markdown("""
            **特征列表：**
            - 时间特征：hourOfDay, dayOfWeek, weekend, holiday
            - 滞后特征：week_X-2, week_X-3, week_X-4
            - 移动平均：MA_X-4
            - 温度特征：T2M_toc
            - 节假日标识：Holiday_ID
            """)

    # 特征重要性说明
    st.info(
        "💡 **使用说明**：选择测试集中的日期，系统会基于该日期前24小时的数据，"
        "预测未来指定时长的电力负荷。图表支持交互式缩放和悬停查看数值。"
    )


if __name__ == "__main__":
    main()
