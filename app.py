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
st.markdown('<p class="sub-title">基于 XGBoost 机器学习模型，上传小时级负荷数据即可获得未来24小时预测</p>', unsafe_allow_html=True)

# -------------------- 加载模型 --------------------
@st.cache_resource
def load_xgb_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "xgb_hourly_load_model.joblib")
    feature_path = os.path.join(script_dir, "feature_names.txt")
    
    try:
        model = joblib.load(model_path)
        with open(feature_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        st.success(f"✅ XGBoost 模型加载成功！特征数: {len(feature_names)}")
        return model, feature_names
    except FileNotFoundError:
        st.error(f"❌ 模型文件未找到，请确保 'xgb_hourly_load_model.joblib' 和 'feature_names.txt' 存在于应用目录")
        return None, None
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None, None

# -------------------- 数据补全（分钟→小时重采样） --------------------
def resample_to_hourly(df, time_col, value_col):
    """
    将分钟级数据重采样为小时级（取平均值）
    """
    try:
        df_hourly = df.copy()
        df_hourly[time_col] = pd.to_datetime(df_hourly[time_col])
        df_hourly = df_hourly.set_index(time_col)
        df_hourly = df_hourly.resample('1h').mean()
        df_hourly = df_hourly.dropna(subset=[value_col])
        df_hourly = df_hourly.reset_index()
        return df_hourly
    except Exception as e:
        raise ValueError(f"数据重采样失败: {str(e)}")

# -------------------- 特征构造（与训练时严格一致） --------------------
def prepare_features(df_hourly, feature_names):
    """
    构造预测所需的全部特征，必须与训练时完全一致
    """
    data = df_hourly.copy()
    data = data.sort_values('datetime').reset_index(drop=True)
    
    # 滞后特征
    data['lag_1'] = data['load'].shift(1)
    data['lag_2'] = data['load'].shift(2)
    data['lag_3'] = data['load'].shift(3)
    data['lag_24'] = data['load'].shift(24)
    data['lag_48'] = data['load'].shift(48)
    data['lag_72'] = data['load'].shift(72)
    data['lag_168'] = data['load'].shift(168)
    
    # 滚动统计
    data['rolling_mean_6'] = data['load'].rolling(6).mean()
    data['rolling_std_6'] = data['load'].rolling(6).std()
    data['rolling_mean_12'] = data['load'].rolling(12).mean()
    data['rolling_std_12'] = data['load'].rolling(12).std()
    data['rolling_mean_24'] = data['load'].rolling(24).mean()
    
    # 时间特征
    data['hour'] = data['datetime'].dt.hour
    data['dayofweek'] = data['datetime'].dt.dayofweek
    data['month'] = data['datetime'].dt.month
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['sin_weekday'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
    data['cos_weekday'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
    
    # 差分特征（提升周期识别能力）
    data['diff_24'] = data['load'] - data['lag_24']
    data['diff_168'] = data['load'] - data['lag_168']
    data['diff_mean_24'] = data['load'] - data['rolling_mean_24']
    
    # 删除 NaN
    data = data.dropna()
    
    # 按训练时的特征顺序提取
    X = data[feature_names]
    return X, data['datetime']
def predict_future_24h(model, df_hourly, feature_names):
    """
    基于最后一段历史数据，滚动预测未来24小时
    返回: future_times (24个时间戳), future_loads (24个预测值)
    """
    # 1. 取最近的历史数据（至少需要168小时 + 24小时缓冲，我们取最近200小时确保够用）
    hist_data = df_hourly.copy()
    hist_data = hist_data.sort_values('datetime').reset_index(drop=True)
    
    # 如果数据太多，只取最近200小时（保证计算速度，同时满足168窗口）
    if len(hist_data) > 200:
        hist_data = hist_data.iloc[-200:].reset_index(drop=True)
    
    # 提取历史负荷列表（用于滚动更新）
    history_loads = hist_data['load'].tolist()
    last_time = hist_data['datetime'].iloc[-1]
    
    # 存储未来预测结果
    future_times = []
    future_loads = []
    
    # 2. 循环预测24步
    for step in range(1, 25):
        # 当前要预测的时刻
        pred_time = last_time + timedelta(hours=step)
        future_times.append(pred_time)
        
        # ---- 构造当前时刻的特征（必须与训练时完全一致） ----
        # 准备一个字典来存放特征值
        feat_dict = {}
        
        # 滞后特征：从 history_loads 中取最后几个值
        # 注意：history_loads 包含了历史真实值 + 之前步骤预测的值
        feat_dict['lag_1'] = history_loads[-1] if len(history_loads) >= 1 else np.nan
        feat_dict['lag_2'] = history_loads[-2] if len(history_loads) >= 2 else np.nan
        feat_dict['lag_3'] = history_loads[-3] if len(history_loads) >= 3 else np.nan
        feat_dict['lag_24'] = history_loads[-24] if len(history_loads) >= 24 else np.nan
        feat_dict['lag_48'] = history_loads[-48] if len(history_loads) >= 48 else np.nan
        feat_dict['lag_72'] = history_loads[-72] if len(history_loads) >= 72 else np.nan
        feat_dict['lag_168'] = history_loads[-168] if len(history_loads) >= 168 else np.nan
        
        # 滚动统计特征（基于当前最新的 history_loads）
        feat_dict['rolling_mean_6'] = np.mean(history_loads[-6:]) if len(history_loads) >= 6 else np.nan
        feat_dict['rolling_std_6'] = np.std(history_loads[-6:]) if len(history_loads) >= 6 else np.nan
        feat_dict['rolling_mean_12'] = np.mean(history_loads[-12:]) if len(history_loads) >= 12 else np.nan
        feat_dict['rolling_std_12'] = np.std(history_loads[-12:]) if len(history_loads) >= 12 else np.nan
        feat_dict['rolling_mean_24'] = np.mean(history_loads[-24:]) if len(history_loads) >= 24 else np.nan
        
        # 时间特征（基于预测时刻 pred_time）
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
        
        # 差分特征
        feat_dict['diff_24'] = feat_dict['lag_1'] - feat_dict['lag_24']  # 当前变化 vs 昨天
        feat_dict['diff_168'] = feat_dict['lag_1'] - feat_dict['lag_168']  # 当前变化 vs 上周
        feat_dict['diff_mean_24'] = feat_dict['lag_1'] - feat_dict['rolling_mean_24']
        
        # 按训练时的特征顺序构造 DataFrame
        X_pred = pd.DataFrame([feat_dict])[feature_names]
        
        # 预测
        pred_val = model.predict(X_pred)[0]
        future_loads.append(pred_val)
        
        # 【关键】：将预测值加入历史列表，供下一步使用（滚动更新）
        history_loads.append(pred_val)
    
    return future_times, future_loads
# -------------------- 主界面 --------------------
def main():
    # 初始化中文字体
    setup_chinese_font()
    
    # 加载模型
    model, feature_names = load_xgb_model()
    if model is None:
        st.stop()
    
    # 侧边栏信息
    st.sidebar.markdown("### 📌 模型信息")
    st.sidebar.info(
        f"模型类型: XGBoost\n"
        f"特征数量: {len(feature_names)}\n"
        f"预测步长: 24 小时\n"
        f"输入要求: 至少 7 天（168小时）历史数据"
    )
    
    # 上传数据
    uploaded_file = st.file_uploader("📂 点击上传或拖拽 CSV 文件（需包含「时间」和「负荷」列）", type=["csv"])
    
    if uploaded_file is not None:
        # 读取数据
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
        
        # 重采样为小时级
        try:
            df_hourly = resample_to_hourly(df, 'datetime', 'load')
            st.success(f"✅ 数据已重采样为小时级，共 {len(df_hourly)} 条记录")
        except Exception as e:
            st.error(f"❌ 重采样失败: {e}")
            return
        
        # 数据概览
        st.subheader("📊 数据概览")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("数据总时长", f"{len(df_hourly)} 小时")
        col_b.metric("起始时间", df_hourly['datetime'].min().strftime('%Y-%m-%d %H:%M'))
        col_c.metric("结束时间", df_hourly['datetime'].max().strftime('%Y-%m-%d %H:%M'))
        
        # 历史曲线
        fig_hist, ax_hist = plt.subplots(figsize=(12, 3))
        ax_hist.plot(df_hourly['datetime'], df_hourly['load'], linewidth=0.8, color='#1E88E5')
        ax_hist.set_title("历史负荷曲线（小时级）")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax_hist.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax_hist.xaxis.get_majorticklabels(), rotation=45, ha='right')
        st.pyplot(fig_hist)
        
        # 预测按钮
                # ===== 替换原来的预测逻辑 =====
        if st.button("🚀 开始智能预测", type="primary", use_container_width=True):
            # 检查数据长度
            MIN_REQUIRED = 168
            if len(df_hourly) < MIN_REQUIRED:
                st.error(f"❌ 数据量不足。需要至少 {MIN_REQUIRED} 小时（7天）历史数据，当前只有 {len(df_hourly)} 小时。")
                return
            
            # 调用滚动预测函数
            try:
                future_times, future_loads = predict_future_24h(model, df_hourly, feature_names)
            except Exception as e:
                st.error(f"❌ 预测失败: {e}")
                return
            
            # 显示预测结果
            st.subheader("🔮 未来24小时预测结果")
            
            # 预测概览卡片
            st.markdown(f"""
            <div class="prediction-box">
                <h4 style="margin-top:0;">📈 未来 24 小时预测概览</h4>
                <table style="width:100%;">
                    <tr><td><b>起始预测时刻</b></td><td>{future_times[0].strftime('%Y-%m-%d %H:%M')}</td>
                    <td><b>峰值负荷</b></td><td style="color:#d32f2f; font-weight:bold;">{max(future_loads):.2f}</td></tr>
                    <tr><td><b>结束预测时刻</b></td><td>{future_times[-1].strftime('%Y-%m-%d %H:%M')}</td>
                    <td><b>平均负荷</b></td><td style="color:#1976d2; font-weight:bold;">{np.mean(future_loads):.2f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # 绘制预测曲线（展示最近3天历史 + 未来24小时）
            fig_pred, ax_pred = plt.subplots(figsize=(14, 5))
            
            # 取最近72小时历史数据（3天）
            show_hist = min(72, len(df_hourly))
            plot_hist_df = df_hourly.iloc[-show_hist:]
            
            ax_pred.plot(plot_hist_df['datetime'], plot_hist_df['load'], 
                        label='历史负荷', linewidth=2, color='#1E88E5')
            ax_pred.plot(future_times, future_loads, 
                        label='未来预测 (XGBoost)', linewidth=2.5, color='#FF6F00', marker='o', markersize=4)
            
            # 在历史与未来交界处画一条竖线
            last_time = df_hourly['datetime'].iloc[-1]
            ax_pred.axvline(x=last_time, color='red', linestyle='--', linewidth=1.5, 
                           label='当前时刻（预测起点）')
            
            ax_pred.legend(fontsize=12)
            ax_pred.set_title("未来24小时负荷走势预测", fontsize=16)
            ax_pred.grid(True, alpha=0.3)
            ax_pred.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax_pred.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax_pred.xaxis.get_majorticklabels(), rotation=45, ha='right')
            st.pyplot(fig_pred)
            
            # 详细表格与下载
            with st.expander("📋 查看详细预测数据表格"):
                result_df = pd.DataFrame({
                    '预测时间': future_times,
                    '预测负荷': future_loads
                })
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
