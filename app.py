import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import joblib
import io
import warnings
import os
from datetime import timedelta

warnings.filterwarnings('ignore')

# -------------------- 中文字体配置 --------------------
@st.cache_resource
def setup_chinese_font():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, 'custom_font.ttf')
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name(), 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        return False

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="智能用能负荷预测系统 (XGBoost)", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        .main-title { font-size: 3rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 0; }
        .sub-title { font-size: 1.2rem; color: #666; text-align: center; margin-top: 0; }
        .prediction-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<p class="main-title">⚡ 智能用能负荷预测系统</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于 XGBoost 机器学习模型，上传 5 分钟级负荷数据即可获得未来24小时预测</p>', unsafe_allow_html=True)

# -------------------- 模型加载（根据类型） --------------------
@st.cache_resource
def load_xgb_model(model_type):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_type == "负荷直接预测":
        model_path = os.path.join(script_dir, "model_xn_xgb1.joblib")
        feature_path = os.path.join(script_dir, "feature_names1.txt")
    else:  # 周偏差预测
        model_path = os.path.join(script_dir, "model_xn_xgb.joblib")
        feature_path = os.path.join(script_dir, "feature_names.txt")
    try:
        model = joblib.load(model_path)
        with open(feature_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        st.success(f"✅ {model_type} 模型加载成功！特征数: {len(feature_names)}")
        return model, feature_names
    except FileNotFoundError:
        st.error(f"❌ 模型文件未找到，请确保 {os.path.basename(model_path)} 和 {os.path.basename(feature_path)} 存在于应用目录")
        return None, None
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None, None

# -------------------- 数据重采样为5分钟 --------------------
def resample_to_5min(df, time_col, value_col):
    """
    将数据重采样为5分钟间隔（取均值）
    """
    try:
        df_5min = df.copy()
        df_5min[time_col] = pd.to_datetime(df_5min[time_col])
        df_5min = df_5min.set_index(time_col)
        df_5min = df_5min.resample('5min').mean()
        df_5min = df_5min.dropna(subset=[value_col])
        df_5min = df_5min.reset_index()
        return df_5min
    except Exception as e:
        raise ValueError(f"数据重采样失败: {str(e)}")

# -------------------- 回测函数（负荷直接预测） --------------------
def backtest_load(model, df_5min, feature_names, test_steps=288, min_history=2016):
    """
    负荷直接预测模型的滚动回测（5分钟步长）
    """
    data = df_5min.copy().sort_values('datetime').reset_index(drop=True)
    if len(data) < min_history + test_steps:
        return None
    test_start_idx = len(data) - test_steps
    history = data.iloc[:test_start_idx].copy()
    test = data.iloc[test_start_idx:].copy()
    history_loads = history['load'].tolist()
    preds = []
    for i in range(test_steps):
        pred_time = test['datetime'].iloc[i]
        # 构造特征字典（与训练时完全一致）
        feat_dict = {}
        # 滞后特征
        feat_dict['lag_1'] = history_loads[-1] if len(history_loads) >= 1 else np.nan
        feat_dict['lag_2'] = history_loads[-2] if len(history_loads) >= 2 else np.nan
        feat_dict['lag_3'] = history_loads[-3] if len(history_loads) >= 3 else np.nan
        feat_dict['lag_24'] = history_loads[-288] if len(history_loads) >= 288 else np.nan
        feat_dict['lag_168'] = history_loads[-2016] if len(history_loads) >= 2016 else np.nan
        # 滚动统计
        feat_dict['rolling_mean_6'] = np.mean(history_loads[-72:]) if len(history_loads) >= 72 else np.nan
        feat_dict['rolling_std_6'] = np.std(history_loads[-72:]) if len(history_loads) >= 72 else np.nan
        feat_dict['rolling_mean_12'] = np.mean(history_loads[-144:]) if len(history_loads) >= 144 else np.nan
        feat_dict['rolling_std_12'] = np.std(history_loads[-144:]) if len(history_loads) >= 144 else np.nan
        feat_dict['rolling_mean_24'] = np.mean(history_loads[-288:]) if len(history_loads) >= 288 else np.nan
        # 时间特征
        hour = pred_time.hour
        dayofweek = pred_time.weekday()
        month = pred_time.month
        feat_dict['hour'] = hour
        feat_dict['hour_lag24'] = hour * feat_dict['lag_24']
        feat_dict['diff_lag24_sq'] = (feat_dict['lag_1'] - feat_dict['lag_24']) ** 2
        feat_dict['dayofweek'] = dayofweek
        feat_dict['month'] = month
        feat_dict['is_weekend'] = 1 if dayofweek >= 5 else 0
        feat_dict['sin_hour'] = np.sin(2 * np.pi * hour / 24)
        feat_dict['cos_hour'] = np.cos(2 * np.pi * hour / 24)
        feat_dict['sin_weekday'] = np.sin(2 * np.pi * dayofweek / 7)
        feat_dict['cos_weekday'] = np.cos(2 * np.pi * dayofweek / 7)
        # 差分特征
        feat_dict['diff_1'] = feat_dict['lag_1'] - feat_dict['lag_2']
        feat_dict['diff_24'] = feat_dict['lag_24'] - history_loads[-289] if len(history_loads) >= 289 else np.nan
        feat_dict['diff_168'] = feat_dict['lag_168'] - history_loads[-2017] if len(history_loads) >= 2017 else np.nan
        # 构造成DataFrame
        X_pred = pd.DataFrame([feat_dict])[feature_names]
        pred_val = model.predict(X_pred)[0]
        preds.append(pred_val)
        history_loads.append(test['load'].iloc[i])  # 使用真实值
    trues = test['load'].values
    preds = np.array(preds)
    mae = np.mean(np.abs(trues - preds))
    mape = np.mean(np.abs((trues - preds) / (trues + 1e-8))) * 100
    rmse = np.sqrt(np.mean((trues - preds) ** 2))
    r2 = 1 - np.sum((trues - preds) ** 2) / np.sum((trues - np.mean(trues)) ** 2)
    return {
        'true': trues,
        'pred': preds,
        'datetime': test['datetime'].values,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'r2': r2
    }

# -------------------- 回测函数（周偏差预测） --------------------
def backtest_diff(model, df_5min, feature_names, test_steps=288, min_history=2016):
    """
    周偏差预测模型的滚动回测（5分钟步长）
    """
    data = df_5min.copy().sort_values('datetime').reset_index(drop=True)
    if len(data) < min_history + test_steps:
        return None
    test_start_idx = len(data) - test_steps
    history = data.iloc[:test_start_idx].copy()
    test = data.iloc[test_start_idx:].copy()
    history_loads = history['load'].tolist()
    # 初始化历史target列表（用于极值特征）
    history_targets = []
    for i in range(len(history_loads)):
        if i >= 2016:
            target_i = history_loads[i] - history_loads[i-2016]
        else:
            target_i = 0.0
        history_targets.append(target_i)
    preds = []
    for i in range(test_steps):
        pred_time = test['datetime'].iloc[i]
        # 构造特征字典（与训练时create_features_on_segment一致）
        feat_dict = {}
        # week_base
        if len(history_loads) >= 2016:
            feat_dict['week_base'] = history_loads[-2016]
        else:
            raise ValueError("历史数据不足以构造 week_base")
        # week_base_rolling
        if len(history_loads) >= 2016 + 6:
            window_start = -2016 - 6
            window_end = -2016 + 6
            feat_dict['week_base_rolling'] = np.mean(history_loads[window_start:window_end])
        else:
            feat_dict['week_base_rolling'] = feat_dict['week_base']
        # delta_24h
        if len(history_loads) >= 288:
            feat_dict['delta_24h'] = history_loads[-1] - history_loads[-288]
        else:
            feat_dict['delta_24h'] = 0.0
        # trend_6h
        if len(history_loads) >= 72:
            feat_dict['trend_6h'] = history_loads[-1] - history_loads[-72]
        else:
            feat_dict['trend_6h'] = 0.0
        # lag_1_target
        if len(history_targets) >= 1:
            feat_dict['lag_1_target'] = history_targets[-1]
        else:
            feat_dict['lag_1_target'] = 0.0
        # lag_2_target
        if len(history_targets) >= 2:
            feat_dict['lag_2_target'] = history_targets[-2]
        else:
            feat_dict['lag_2_target'] = 0.0
        # lag_6h_target
        if len(history_targets) >= 72:
            feat_dict['lag_6h_target'] = history_targets[-72]
        else:
            feat_dict['lag_6h_target'] = 0.0
        # 极值特征（过去6小时）
        if len(history_targets) >= 73:
            past = history_targets[-73:-1]
        else:
            past = history_targets[:-1]
        if len(past) > 0:
            feat_dict['max_6h_past'] = np.max(past)
            feat_dict['min_6h_past'] = np.min(past)
            feat_dict['range_6h_past'] = feat_dict['max_6h_past'] - feat_dict['min_6h_past']
            feat_dict['mean_6h_past'] = np.mean(past)
        else:
            feat_dict['max_6h_past'] = 0.0
            feat_dict['min_6h_past'] = 0.0
            feat_dict['range_6h_past'] = 0.0
            feat_dict['mean_6h_past'] = 0.0
        # 过去1小时
        if len(history_targets) >= 13:
            past = history_targets[-13:-1]
        else:
            past = history_targets[:-1]
        if len(past) > 0:
            feat_dict['max_1h_past'] = np.max(past)
            feat_dict['min_1h_past'] = np.min(past)
            feat_dict['range_1h_past'] = feat_dict['max_1h_past'] - feat_dict['min_1h_past']
            feat_dict['mean_1h_past'] = np.mean(past)
        else:
            feat_dict['max_1h_past'] = 0.0
            feat_dict['min_1h_past'] = 0.0
            feat_dict['range_1h_past'] = 0.0
            feat_dict['mean_1h_past'] = 0.0
        # 过去0.5小时
        if len(history_targets) >= 7:
            past = history_targets[-7:-1]
        else:
            past = history_targets[:-1]
        if len(past) > 0:
            feat_dict['max_half_hour_past'] = np.max(past)
            feat_dict['min_half_hour_past'] = np.min(past)
            feat_dict['range_half_hour_past'] = feat_dict['max_half_hour_past'] - feat_dict['min_half_hour_past']
            feat_dict['mean_half_hour_past'] = np.mean(past)
        else:
            feat_dict['max_half_hour_past'] = 0.0
            feat_dict['min_half_hour_past'] = 0.0
            feat_dict['range_half_hour_past'] = 0.0
            feat_dict['mean_half_hour_past'] = 0.0
        # 过去1天
        if len(history_targets) >= 289:
            past = history_targets[-289:-1]
        else:
            past = history_targets[:-1]
        if len(past) > 0:
            feat_dict['max_1d_past'] = np.max(past)
            feat_dict['min_1d_past'] = np.min(past)
            feat_dict['range_1d_past'] = feat_dict['max_1d_past'] - feat_dict['min_1d_past']
            feat_dict['mean_1d_past'] = np.mean(past)
        else:
            feat_dict['max_1d_past'] = 0.0
            feat_dict['min_1d_past'] = 0.0
            feat_dict['range_1d_past'] = 0.0
            feat_dict['mean_1d_past'] = 0.0
        # 时间特征
        hour = pred_time.hour
        dayofweek = pred_time.weekday()
        month = pred_time.month
        feat_dict['hour'] = hour
        feat_dict['dayofweek'] = dayofweek
        feat_dict['month'] = month
        feat_dict['is_weekend'] = 1 if dayofweek >= 5 else 0
        feat_dict['sin_hour'] = np.sin(2 * np.pi * hour / 24)
        feat_dict['cos_hour'] = np.cos(2 * np.pi * hour / 24)
        feat_dict['sin_weekday'] = np.sin(2 * np.pi * dayofweek / 7)
        feat_dict['cos_weekday'] = np.cos(2 * np.pi * dayofweek / 7)
        # 构造成DataFrame
        X_pred = pd.DataFrame([feat_dict])[feature_names]
        delta_pred = model.predict(X_pred)[0]
        load_pred = feat_dict['week_base'] + delta_pred
        preds.append(load_pred)
        history_loads.append(test['load'].iloc[i])  # 真实值用于驱动
        # 更新history_targets
        new_target = test['load'].iloc[i] - (history_loads[-2017] if len(history_loads) >= 2017 else 0.0)
        history_targets.append(new_target)
    trues = test['load'].values
    preds = np.array(preds)
    mae = np.mean(np.abs(trues - preds))
    mape = np.mean(np.abs((trues - preds) / (trues + 1e-8))) * 100
    rmse = np.sqrt(np.mean((trues - preds) ** 2))
    r2 = 1 - np.sum((trues - preds) ** 2) / np.sum((trues - np.mean(trues)) ** 2)
    return {
        'true': trues,
        'pred': preds,
        'datetime': test['datetime'].values,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'r2': r2
    }

# -------------------- 预测未来24小时（负荷直接预测） --------------------
def predict_future_24h_load(model, df_5min, feature_names):
    """
    滚动预测未来24小时（288个5分钟点），负荷直接预测模型
    """
    hist_data = df_5min.copy().sort_values('datetime').reset_index(drop=True)
    if len(hist_data) > 2000:
        hist_data = hist_data.iloc[-2000:].reset_index(drop=True)
    history_loads = hist_data['load'].tolist()
    last_time = hist_data['datetime'].iloc[-1]
    steps = 288  # 24小时 * 12点/小时
    future_times = []
    future_loads = []
    for step in range(1, steps + 1):
        pred_time = last_time + timedelta(minutes=5 * step)
        future_times.append(pred_time)
        feat_dict = {}
        # 滞后特征
        feat_dict['lag_1'] = history_loads[-1] if len(history_loads) >= 1 else np.nan
        feat_dict['lag_2'] = history_loads[-2] if len(history_loads) >= 2 else np.nan
        feat_dict['lag_3'] = history_loads[-3] if len(history_loads) >= 3 else np.nan
        feat_dict['lag_24'] = history_loads[-288] if len(history_loads) >= 288 else np.nan
        feat_dict['lag_168'] = history_loads[-2016] if len(history_loads) >= 2016 else np.nan
        # 滚动统计
        feat_dict['rolling_mean_6'] = np.mean(history_loads[-72:]) if len(history_loads) >= 72 else np.nan
        feat_dict['rolling_std_6'] = np.std(history_loads[-72:]) if len(history_loads) >= 72 else np.nan
        feat_dict['rolling_mean_12'] = np.mean(history_loads[-144:]) if len(history_loads) >= 144 else np.nan
        feat_dict['rolling_std_12'] = np.std(history_loads[-144:]) if len(history_loads) >= 144 else np.nan
        feat_dict['rolling_mean_24'] = np.mean(history_loads[-288:]) if len(history_loads) >= 288 else np.nan
        # 时间特征
        hour = pred_time.hour
        dayofweek = pred_time.weekday()
        month = pred_time.month
        feat_dict['hour'] = hour
        feat_dict['hour_lag24'] = hour * feat_dict['lag_24']
        feat_dict['diff_lag24_sq'] = (feat_dict['lag_1'] - feat_dict['lag_24']) ** 2
        feat_dict['dayofweek'] = dayofweek
        feat_dict['month'] = month
        feat_dict['is_weekend'] = 1 if dayofweek >= 5 else 0
        feat_dict['sin_hour'] = np.sin(2 * np.pi * hour / 24)
        feat_dict['cos_hour'] = np.cos(2 * np.pi * hour / 24)
        feat_dict['sin_weekday'] = np.sin(2 * np.pi * dayofweek / 7)
        feat_dict['cos_weekday'] = np.cos(2 * np.pi * dayofweek / 7)
        # 差分特征
        feat_dict['diff_1'] = feat_dict['lag_1'] - feat_dict['lag_2']
        feat_dict['diff_24'] = feat_dict['lag_24'] - history_loads[-289] if len(history_loads) >= 289 else np.nan
        feat_dict['diff_168'] = feat_dict['lag_168'] - history_loads[-2017] if len(history_loads) >= 2017 else np.nan
        X_pred = pd.DataFrame([feat_dict])[feature_names]
        pred_val = model.predict(X_pred)[0]
        future_loads.append(pred_val)
        history_loads.append(pred_val)
    return future_times, future_loads

# -------------------- 预测未来24小时（周偏差预测） --------------------
def predict_future_24h_diff(model, df_5min, feature_names):
    """
    滚动预测未来24小时（288个5分钟点），周偏差预测模型
    """
    hist_data = df_5min.copy().sort_values('datetime').reset_index(drop=True)
    if len(hist_data) > 2500:
        hist_data = hist_data.iloc[-2500:].reset_index(drop=True)
    history_loads = hist_data['load'].tolist()
    # 初始化history_targets
    history_targets = []
    for i in range(len(history_loads)):
        if i >= 2016:
            target_i = history_loads[i] - history_loads[i-2016]
        else:
            target_i = 0.0
        history_targets.append(target_i)
    last_time = hist_data['datetime'].iloc[-1]
    steps = 288
    future_times = []
    future_loads = []
    for step in range(1, steps + 1):
        pred_time = last_time + timedelta(minutes=5 * step)
        future_times.append(pred_time)
        feat_dict = {}
        # week_base
        if len(history_loads) >= 2016:
            feat_dict['week_base'] = history_loads[-2016]
        else:
            raise ValueError("历史数据不足")
        # week_base_rolling
        if len(history_loads) >= 2016 + 6:
            window_start = -2016 - 6
            window_end = -2016 + 6
            feat_dict['week_base_rolling'] = np.mean(history_loads[window_start:window_end])
        else:
            feat_dict['week_base_rolling'] = feat_dict['week_base']
        # delta_24h
        if len(history_loads) >= 288:
            feat_dict['delta_24h'] = history_loads[-1] - history_loads[-288]
        else:
            feat_dict['delta_24h'] = 0.0
        # trend_6h
        if len(history_loads) >= 72:
            feat_dict['trend_6h'] = history_loads[-1] - history_loads[-72]
        else:
            feat_dict['trend_6h'] = 0.0
        # lag_1_target
        if len(history_targets) >= 1:
            feat_dict['lag_1_target'] = history_targets[-1]
        else:
            feat_dict['lag_1_target'] = 0.0
        # lag_2_target
        if len(history_targets) >= 2:
            feat_dict['lag_2_target'] = history_targets[-2]
        else:
            feat_dict['lag_2_target'] = 0.0
        # lag_6h_target
        if len(history_targets) >= 72:
            feat_dict['lag_6h_target'] = history_targets[-72]
        else:
            feat_dict['lag_6h_target'] = 0.0
        # 极值特征
        # 过去6小时
        if len(history_targets) >= 73:
            past = history_targets[-73:-1]
        else:
            past = history_targets[:-1]
        if len(past) > 0:
            feat_dict['max_6h_past'] = np.max(past)
            feat_dict['min_6h_past'] = np.min(past)
            feat_dict['range_6h_past'] = feat_dict['max_6h_past'] - feat_dict['min_6h_past']
            feat_dict['mean_6h_past'] = np.mean(past)
        else:
            feat_dict['max_6h_past'] = 0.0
            feat_dict['min_6h_past'] = 0.0
            feat_dict['range_6h_past'] = 0.0
            feat_dict['mean_6h_past'] = 0.0
        # 过去1小时
        if len(history_targets) >= 13:
            past = history_targets[-13:-1]
        else:
            past = history_targets[:-1]
        if len(past) > 0:
            feat_dict['max_1h_past'] = np.max(past)
            feat_dict['min_1h_past'] = np.min(past)
            feat_dict['range_1h_past'] = feat_dict['max_1h_past'] - feat_dict['min_1h_past']
            feat_dict['mean_1h_past'] = np.mean(past)
        else:
            feat_dict['max_1h_past'] = 0.0
            feat_dict['min_1h_past'] = 0.0
            feat_dict['range_1h_past'] = 0.0
            feat_dict['mean_1h_past'] = 0.0
        # 过去0.5小时
        if len(history_targets) >= 7:
            past = history_targets[-7:-1]
        else:
            past = history_targets[:-1]
        if len(past) > 0:
            feat_dict['max_half_hour_past'] = np.max(past)
            feat_dict['min_half_hour_past'] = np.min(past)
            feat_dict['range_half_hour_past'] = feat_dict['max_half_hour_past'] - feat_dict['min_half_hour_past']
            feat_dict['mean_half_hour_past'] = np.mean(past)
        else:
            feat_dict['max_half_hour_past'] = 0.0
            feat_dict['min_half_hour_past'] = 0.0
            feat_dict['range_half_hour_past'] = 0.0
            feat_dict['mean_half_hour_past'] = 0.0
        # 过去1天
        if len(history_targets) >= 289:
            past = history_targets[-289:-1]
        else:
            past = history_targets[:-1]
        if len(past) > 0:
            feat_dict['max_1d_past'] = np.max(past)
            feat_dict['min_1d_past'] = np.min(past)
            feat_dict['range_1d_past'] = feat_dict['max_1d_past'] - feat_dict['min_1d_past']
            feat_dict['mean_1d_past'] = np.mean(past)
        else:
            feat_dict['max_1d_past'] = 0.0
            feat_dict['min_1d_past'] = 0.0
            feat_dict['range_1d_past'] = 0.0
            feat_dict['mean_1d_past'] = 0.0
        # 时间特征
        hour = pred_time.hour
        dayofweek = pred_time.weekday()
        month = pred_time.month
        feat_dict['hour'] = hour
        feat_dict['dayofweek'] = dayofweek
        feat_dict['month'] = month
        feat_dict['is_weekend'] = 1 if dayofweek >= 5 else 0
        feat_dict['sin_hour'] = np.sin(2 * np.pi * hour / 24)
        feat_dict['cos_hour'] = np.cos(2 * np.pi * hour / 24)
        feat_dict['sin_weekday'] = np.sin(2 * np.pi * dayofweek / 7)
        feat_dict['cos_weekday'] = np.cos(2 * np.pi * dayofweek / 7)
        X_pred = pd.DataFrame([feat_dict])[feature_names]
        delta_pred = model.predict(X_pred)[0]
        load_pred = feat_dict['week_base'] + delta_pred
        future_loads.append(load_pred)
        history_loads.append(load_pred)
        history_targets.append(delta_pred)
    return future_times, future_loads

# -------------------- 主界面 --------------------
def main():
    setup_chinese_font()
    
    # 模型类型选择
    model_type = st.radio("选择模型类型", ("负荷直接预测", "周偏差预测"))
    
    model, feature_names = load_xgb_model(model_type)
    if model is None:
        st.stop()
    
    st.sidebar.markdown("### 📌 模型信息")
    st.sidebar.info(
        f"模型类型: {model_type}\n"
        f"特征数量: {len(feature_names)}\n"
        f"预测步长: 24小时（288个5分钟点）\n"
        f"输入要求: 至少 7 天（2016个点）历史数据"
    )
    
    uploaded_file = st.file_uploader("📂 点击上传或拖拽 CSV 文件（需包含「时间」和「负荷」列）", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # 智能识别列名
        time_col, load_col = None, None
        for col in df.columns:
            if '时间' in col or '日期' in col or 'datetime' in col.lower() or 'timestamp' in col.lower():
                time_col = col
            if '负荷' in col or '功率' in col or 'load' in col.lower() or 'value' in col.lower():
                load_col = col
        if time_col is None:
            time_col = df.columns[0]
        if load_col is None:
            load_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        try:
            df['datetime'] = pd.to_datetime(df[time_col])
            df['load'] = df[load_col].astype(float)
        except:
            st.error("❌ 日期或数值格式解析失败，请检查数据。")
            return
        df = df.dropna(subset=['load'])
        
        # 重采样为5分钟
        try:
            df_5min = resample_to_5min(df, 'datetime', 'load')
            st.success(f"✅ 数据已重采样为5分钟级，共 {len(df_5min)} 条记录")
        except Exception as e:
            st.error(f"❌ 重采样失败: {e}")
            return
        
        # 数据概览
        st.subheader("📊 数据概览")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("数据总时长", f"{len(df_5min)} 个5分钟点")
        col_b.metric("起始时间", df_5min['datetime'].min().strftime('%Y-%m-%d %H:%M'))
        col_c.metric("结束时间", df_5min['datetime'].max().strftime('%Y-%m-%d %H:%M'))
        
        # 历史曲线
        fig_hist, ax_hist = plt.subplots(figsize=(12, 3))
        ax_hist.plot(df_5min['datetime'], df_5min['load'], linewidth=0.8, color='#1E88E5')
        ax_hist.set_title("Historical Load Curve (5-Minute)")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax_hist.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax_hist.xaxis.get_majorticklabels(), rotation=45, ha='right')
        st.pyplot(fig_hist)
        
        # 回测评估
        st.subheader("📊 模型回测评估")
        min_history = 2016
        test_steps = 288
        if len(df_5min) >= min_history + test_steps:
            with st.spinner("正在运行回测评估，请稍候..."):
                if model_type == "负荷直接预测":
                    backtest_result = backtest_load(model, df_5min, feature_names, test_steps, min_history)
                else:
                    backtest_result = backtest_diff(model, df_5min, feature_names, test_steps, min_history)
            if backtest_result is not None:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{backtest_result['mae']:.3f}")
                col2.metric("MAPE", f"{backtest_result['mape']:.2f}%")
                col3.metric("RMSE", f"{backtest_result['rmse']:.3f}")
                col4.metric("R²", f"{backtest_result['r2']:.4f}")
                fig_back, ax_back = plt.subplots(figsize=(14, 4))
                ax_back.plot(backtest_result['datetime'], backtest_result['true'], label='True load', linewidth=2, color='#1E88E5')
                ax_back.plot(backtest_result['datetime'], backtest_result['pred'], label='Predicted load', linewidth=2, linestyle='--', color='#FF6F00')
                ax_back.legend(fontsize=12)
                ax_back.set_title(f"Backtesting Results (the most recent {test_steps} 5-minute data points)", fontsize=14)
                ax_back.grid(True, alpha=0.3)
                ax_back.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                ax_back.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.setp(ax_back.xaxis.get_majorticklabels(), rotation=45, ha='right')
                st.pyplot(fig_back)
            else:
                st.warning("回测失败，数据可能不足。")
        else:
            st.info(f"当前数据量 {len(df_5min)} 个点，需要至少 {min_history + test_steps} 个点才能进行回测评估。")
        
        # 预测按钮
        if st.button("🚀 开始智能预测", type="primary", use_container_width=True):
            if len(df_5min) < min_history:
                st.error(f"❌ 数据量不足。需要至少 {min_history} 个点（7天）历史数据，当前只有 {len(df_5min)} 个点。")
                return
            try:
                if model_type == "负荷直接预测":
                    future_times, future_loads = predict_future_24h_load(model, df_5min, feature_names)
                else:
                    future_times, future_loads = predict_future_24h_diff(model, df_5min, feature_names)
            except Exception as e:
                st.error(f"❌ 预测失败: {e}")
                return
            
            st.subheader("🔮 未来24小时预测结果")
            st.markdown(f"""
            <div class="prediction-box">
                <h4 style="margin-top:0;">📈 未来 24 小时（288个5分钟点）预测概览</h4>
                <table style="width:100%;">
                    <tr><td><b>起始预测时刻</b></td><td>{future_times[0].strftime('%Y-%m-%d %H:%M')}</td>
                    <td><b>峰值负荷</b></td><td style="color:#d32f2f; font-weight:bold;">{max(future_loads):.2f}</td></tr>
                    <tr><td><b>结束预测时刻</b></td><td>{future_times[-1].strftime('%Y-%m-%d %H:%M')}</td>
                    <td><b>平均负荷</b></td><td style="color:#1976d2; font-weight:bold;">{np.mean(future_loads):.2f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            fig_pred, ax_pred = plt.subplots(figsize=(14, 5))
            show_hist = min(288, len(df_5min))  # 展示最近24小时历史
            plot_hist_df = df_5min.iloc[-show_hist:]
            ax_pred.plot(plot_hist_df['datetime'], plot_hist_df['load'], label='Historical Load', linewidth=2, color='#1E88E5')
            # 连接历史和预测
            connected_times = [plot_hist_df['datetime'].iloc[-1]] + future_times
            connected_loads = [plot_hist_df['load'].iloc[-1]] + future_loads
            ax_pred.plot(connected_times, connected_loads, label='Future Forecasts (XGBoost)', linewidth=2.5, color='#FF6F00', marker='o', markersize=3)
            last_time = df_5min['datetime'].iloc[-1]
            ax_pred.axvline(x=last_time, color='red', linestyle='--', linewidth=1.5, label='Current Time (Prediction Start)')
            ax_pred.legend(fontsize=12)
            ax_pred.set_title("Load Forecast for the Next 24 Hours (5-Minute)", fontsize=16)
            ax_pred.grid(True, alpha=0.3)
            ax_pred.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax_pred.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax_pred.xaxis.get_majorticklabels(), rotation=45, ha='right')
            st.pyplot(fig_pred)
            
            with st.expander("📋 查看详细预测数据表格"):
                result_df = pd.DataFrame({'预测时间': future_times, '预测负荷': future_loads})
                st.dataframe(result_df, use_container_width=True)
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 下载未来24小时预测结果 (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"未来24h预测_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
