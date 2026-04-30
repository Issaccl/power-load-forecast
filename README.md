# ⚡ 电力负荷预测系统

基于 LSTM 深度学习的电力负荷预测系统，支持交互式网页可视化。

## 🎯 项目简介

- 使用 PyTorch 构建 LSTM 神经网络
- 预测未来 1-72 小时的电力负荷
- 测试集 MAPE：**3.65%**
- Streamlit 交互式网页部署
- SwanLab 实验跟踪

## 📊 特征工程

- 时间特征：hourOfDay, dayOfWeek, weekend, holiday
- 滞后特征：week_X-2, week_X-3, week_X-4
- 移动平均：MA_X-4
- 温度特征：T2M_toc

## 🏗️ 模型结构

| 参数 | 值 |
|------|-----|
| 模型类型 | LSTM |
| 输入维度 | 10 |
| 隐藏层大小 | 64 |
| LSTM 层数 | 1 |
| 回顾窗口 | 24 小时 |
| 批大小 | 32 |
| 学习率 | 0.001 |
| 训练轮次 | 50 |

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
