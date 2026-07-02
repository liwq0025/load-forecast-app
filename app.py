import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
import io
import warnings
import os
import urllib.request   # 新增，用于下载字体

warnings.filterwarnings('ignore')
# -------------------- 中文字体配置（解决云服务器中文乱码） --------------------
@st.cache_resource
@st.cache_resource
def setup_chinese_font():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, 'custom_font.ttf')   # 你上传的字体文件名
    
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name(), 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    else:
        # 如果找不到，尝试使用系统默认（可能无效，但不会报错）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        return False
    
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

# -------------------- 数据补全函数 --------------------
def fill_missing_minutes(df, time_col, value_col):
    """
    将时间序列补全为连续的分钟级数据，缺失值使用线性插值
    """
    try:
        # 1. 复制数据，避免修改原数据
        df_filled = df.copy()
        
        # 2. 确保时间列是 datetime 类型
        df_filled[time_col] = pd.to_datetime(df_filled[time_col])
        
        # 3. 删除重复时间（保留第一条）
        df_filled = df_filled.drop_duplicates(subset=[time_col], keep='first')
        
        # 4. 按时间排序
        df_filled = df_filled.sort_values(by=time_col)
        
        # 5. 设置时间索引
        df_filled = df_filled.set_index(time_col)
        
        # 6. 按分钟重采样
        df_filled = df_filled.asfreq('1min')
        
        # 7. 线性插值
        df_filled[value_col] = df_filled[value_col].interpolate(
            method='linear', 
            limit_direction='both'
        )
        
        # 8. 重置索引
        df_filled = df_filled.reset_index()
        
        return df_filled
        
    except Exception as e:
        # 如果出错，将错误信息打印并返回原数据（或重新抛出）
        raise ValueError(f"数据补全失败: {str(e)}")
# -------------------- 主界面 --------------------
def main():
    # 初始化中文字体
    setup_chinese_font()
    
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
        
        # try:
        #     df['datetime'] = pd.to_datetime(df[time_col])
        #     df['load'] = df[load_col].astype(float)
        # except:
        #     st.error("❌ 日期或数值格式解析失败，请检查数据。")
        #     return
        # df = df.dropna(subset=['load'])

        try:
            df['datetime'] = pd.to_datetime(df[time_col])
            df['load'] = df[load_col].astype(float)
        except:
            st.error("❌ 日期或数值格式解析失败，请检查数据。")
            return
        df = df.dropna(subset=['load'])
        
        # ===== 新增：数据补全 =====
        # 调用插值函数，将数据补全为连续的分钟级序列
        try:
            df = fill_missing_minutes(df, time_col='datetime', value_col='load')
            st.success(f"✅ 数据已补全为连续分钟序列，共 {len(df)} 条记录")
        except Exception as e:
            st.error(f"❌ 数据补全失败: {e}")
            st.stop()   # 停止后续执行
        
                
        # 数据概览
        st.subheader("📊 数据概览")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("数据总时长", f"{len(df)} 分钟")
        col_b.metric("起始时间", df['datetime'].min().strftime('%Y-%m-%d %H:%M'))
        col_c.metric("结束时间", df['datetime'].max().strftime('%Y-%m-%d %H:%M'))
        
        # fig_hist, ax_hist = plt.subplots(figsize=(12, 3))
        # ax_hist.plot(df['datetime'], df['load'], linewidth=0.8, color='#1E88E5')
        # ax_hist.set_title("Historical data curve")
        # ax_hist.grid(True, alpha=0.3)
        # st.pyplot(fig_hist)
        
        fig_hist, ax_hist = plt.subplots(figsize=(12, 3))
        ax_hist.plot(df['datetime'], df['load'], linewidth=0.8, color='#1E88E5')
        ax_hist.set_title("Historical data curve")
        ax_hist.grid(True, alpha=0.3)
        
        # ----- 新增：横坐标刻度设为每小时一个标签 -----
        ax_hist.xaxis.set_major_locator(mdates.HourLocator(interval=2))   # 每小时一个主刻度
        ax_hist.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))  # 格式：月-日 时:分
        # 如果数据跨度较短（比如只有几天），还可以显示更密集的次刻度：
        # ax_hist.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # 每30分钟一个次刻度
        # 防止标签重叠，旋转45度
        plt.setp(ax_hist.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
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
            
            # # 取最后 INPUT_STEPS 个点作为输入
            # last_seq = load_scaled[-INPUT_STEPS:]
            # input_tensor = torch.FloatTensor(last_seq).unsqueeze(0).unsqueeze(0).to(device)  # shape (1,1,720)
            
            # # 预测
            # with torch.no_grad():
            #     pred_scaled = model(input_tensor).cpu().numpy().flatten()  # shape (60,)

            # 自回归滚动预测：逐点预测，每次用最新的720点输入，只取输出的第一个点
            seq = load_scaled[-INPUT_STEPS:].copy()   # 初始序列，长度为720
            pred_scaled_list = []
            for _ in range(OUTPUT_STEPS):
                input_tensor = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,720)
                with torch.no_grad():
                    # 模型输出60步，但我们只取第一步（即下一分钟的预测）
                    step_pred = model(input_tensor).cpu().numpy().flatten()[0]
                pred_scaled_list.append(step_pred)
                # 滑动窗口：丢弃最旧的一个点，加入新预测值
                seq = np.append(seq[1:], step_pred)
            pred_scaled = np.array(pred_scaled_list)   # shape (60,)
            
            # 反标准化
            pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
            # 生成未来时间戳
            last_time = df['datetime'].iloc[-1]
            # future_times = pd.date_range(start=last_time + pd.Timedelta(minutes=1), periods=OUTPUT_STEPS, freq='1T')
            future_times = pd.date_range(start=last_time + pd.Timedelta(minutes=1), periods=OUTPUT_STEPS, freq='min')
            
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
                        label='Historical data', linewidth=2, color='#1E88E5')
            ax_pred.plot(future_times, pred_original, 
                        label='Predicted data', linewidth=2.0, color='#FF6F00', marker='o', markersize=1)
            ax_pred.axvline(x=last_time, color='red', linestyle='--', linewidth=1.5, label='The current moment (prediction starting point)')
            ax_pred.legend(fontsize=12)
            ax_pred.set_title("Energy consumption trend forecast (next hour)", fontsize=16)
            ax_pred.grid(True, alpha=0.3)
            st.pyplot(fig_pred)
            
            # 详细表格与下载
            with st.expander("📋 查看详细预测数据表格"):
                result_df = pd.DataFrame({'time': future_times, 'load': pred_original})
                st.dataframe(result_df, use_container_width=True)
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(label="📥 下载预测结果 (CSV)", data=csv_buffer.getvalue(),
                                   file_name=f"负荷预测结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        else:
            st.info("👆 请上传您的历史负荷数据 CSV 文件开始预测。")

if __name__ == "__main__":
    main()
