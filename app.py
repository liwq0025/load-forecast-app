import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import warnings
import os
warnings.filterwarnings('ignore')

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="智能用能负荷预测系统", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        .main-title { font-size: 3rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 0; }
        .sub-title { font-size: 1.2rem; color: #666; text-align: center; margin-top: 0; }
        .prediction-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<p class="main-title">⚡ 智能用能负荷预测系统</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于预训练深度时间卷积网络（TCN），上传数据即可获得预测结果</p>', unsafe_allow_html=True)

# -------------------- TCN 核心网络（必须与训练时完全一致） --------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1]
        out = self.linear(out)
        return out

# -------------------- 加载预训练模型（缓存，只加载一次） --------------------
@st.cache_resource
def load_model(device='cpu'):
    # 获取当前 app.py 文件所在的文件夹路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出模型文件的完整绝对路径
    model_path = os.path.join(script_dir, "tcn_load_forecast_best1.pth")
    # 注意：这些参数必须与训练时完全一致！
    INPUT_SIZE = 1
    OUTPUT_STEPS = 60       # 预测未来60分钟
    NUM_CHANNELS = [32, 64, 128]
    KERNEL_SIZE = 5
    DROPOUT = 0.3
    
    model = TCN(input_size=INPUT_SIZE, output_size=OUTPUT_STEPS,
                num_channels=NUM_CHANNELS, kernel_size=KERNEL_SIZE, dropout=DROPOUT)
    
    # 加载权重
    try:
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        # 兼容不同保存方式：如果保存的是整个字典，取model_state_dict；如果是直接state_dict，则直接加载
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        st.success("✅ 预训练模型加载成功！")
        return model, OUTPUT_STEPS
    except Exception as e:
        st.error(f"❌ 模型加载失败，请确认模型文件存在且格式正确。错误信息：{e}")
        return None, None

# -------------------- 主界面 --------------------
def main():
    # 固定模型参数（显示给用户，方便知道预测步长）
    st.sidebar.markdown("### 📌 模型信息")
    st.sidebar.info("预训练模型参数：\n- 输入：过去720分钟\n- 输出：未来60分钟")
    
    # 加载模型（默认在当前目录寻找 tcn_load_forecast_best1.pth）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model, OUTPUT_STEPS = load_model("tcn_load_forecast_best1.pth", device)
    model, OUTPUT_STEPS = load_model(device)
    if model is None:
        st.stop()   # 如果模型加载失败，停止后续操作
    
    # 上传数据
    uploaded_file = st.file_uploader("📂 点击上传或拖拽 CSV 文件（需包含「时间」和「负荷」列）", type=["csv"])
    
    if uploaded_file is not None:
        # 读取数据，智能识别列名（与之前相同）
        df = pd.read_csv(uploaded_file, parse_dates=[0])
        time_col, load_col = None, None
        for col in df.columns:
            if '时间' in col or '日期' in col or 'datetime' in col.lower():
                time_col = col
            if '负荷' in col or '功率' in col or 'load' in col.lower():
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
        
        # 数据概览
        st.subheader("📊 数据概览")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("数据总时长", f"{len(df)} 分钟")
        col_b.metric("起始时间", df['datetime'].min().strftime('%Y-%m-%d %H:%M'))
        col_c.metric("结束时间", df['datetime'].max().strftime('%Y-%m-%d %H:%M'))
        
        fig_hist, ax_hist = plt.subplots(figsize=(12, 3))
        ax_hist.plot(df['datetime'], df['load'], linewidth=0.8, color='#1E88E5')
        ax_hist.set_title("历史负荷曲线")
        ax_hist.grid(True, alpha=0.3)
        st.pyplot(fig_hist)
        
        # 预测按钮
        if st.button("🚀 开始智能预测", type="primary", use_container_width=True):
            # 检查数据长度是否满足模型要求（至少需要 INPUT_STEPS 条）
            INPUT_STEPS = 720   # 与训练时一致
            if len(df) < INPUT_STEPS:
                st.error(f"❌ 数据量不足。模型需要至少 {INPUT_STEPS} 条历史数据，当前只有 {len(df)} 条。")
                return
            
            # 标准化（使用与训练时相同的Scaler逻辑，但使用当前数据的均值和标准差）
            scaler = StandardScaler()
            load_scaled = scaler.fit_transform(df['load'].values.reshape(-1, 1)).flatten()
            
            # 取最后 INPUT_STEPS 个点作为输入
            last_seq = load_scaled[-INPUT_STEPS:]
            input_tensor = torch.FloatTensor(last_seq).unsqueeze(0).unsqueeze(0).to(device)  # shape (1,1,720)
            
            # 预测
            with torch.no_grad():
                pred_scaled = model(input_tensor).cpu().numpy().flatten()  # shape (60,)
            
            # 反标准化
            pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
            # 生成未来时间戳
            last_time = df['datetime'].iloc[-1]
            future_times = pd.date_range(start=last_time + pd.Timedelta(minutes=1), periods=OUTPUT_STEPS, freq='1T')
            
            # 显示预测结果
            st.subheader("🔮 预测结果分析")
            st.markdown(f"""
            <div class="prediction-box">
                <h4 style="margin-top:0;">📈 未来 {OUTPUT_STEPS} 分钟预测概览</h4>
                <table style="width:100%;">
                    <tr><td><b>起始预测时刻</b></td><td>{future_times[0].strftime('%Y-%m-%d %H:%M')}</td>
                    <td><b>峰值负荷</b></td><td style="color:#d32f2f; font-weight:bold;">{pred_original.max():.2f}</td></tr>
                    <tr><td><b>结束预测时刻</b></td><td>{future_times[-1].strftime('%Y-%m-%d %H:%M')}</td>
                    <td><b>平均负荷</b></td><td style="color:#1976d2; font-weight:bold;">{pred_original.mean():.2f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # 绘制预测曲线
            fig_pred, ax_pred = plt.subplots(figsize=(14, 5))
            show_hist = min(240, len(df))
            plot_hist_df = df.iloc[-show_hist:]
            ax_pred.plot(plot_hist_df['datetime'], plot_hist_df['load'], 
                        label='历史负荷', linewidth=2, color='#1E88E5')
            ax_pred.plot(future_times, pred_original, 
                        label='预测负荷 (AI生成)', linewidth=2.5, color='#FF6F00', marker='o', markersize=5)
            ax_pred.axvline(x=last_time, color='red', linestyle='--', linewidth=1.5, label='当前时刻（预测起点）')
            ax_pred.legend(fontsize=12)
            ax_pred.set_title("用能趋势预测（未来一小时）", fontsize=16)
            ax_pred.grid(True, alpha=0.3)
            st.pyplot(fig_pred)
            
            # 详细表格与下载
            with st.expander("📋 查看详细预测数据表格"):
                result_df = pd.DataFrame({'预测时间点': future_times, '预测负荷值': pred_original})
                st.dataframe(result_df, use_container_width=True)
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(label="📥 下载预测结果 (CSV)", data=csv_buffer.getvalue(),
                                   file_name=f"负荷预测结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        else:
            st.info("👆 请上传您的历史负荷数据 CSV 文件开始预测。")

if __name__ == "__main__":
    main()
