import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import io
import warnings
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
st.markdown('<p class="sub-title">基于深度时间卷积网络（TCN），上传历史负荷数据即可自动预测未来趋势</p>', unsafe_allow_html=True)

# -------------------- TCN 核心网络 --------------------
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

# -------------------- 辅助函数 --------------------
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)

def train_model(model, train_loader, val_loader, epochs, lr, device, progress_bar):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        progress_bar.progress((epoch + 1) / epochs, text=f"训练中 Epoch {epoch+1}/{epochs} (验证损失: {avg_val_loss:.4f})")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
    return best_state

# -------------------- 主界面 --------------------
def main():
    with st.expander("⚙️ 高级参数设置（非必要请勿修改）"):
        col1, col2 = st.columns(2)
        with col1:
            manual_input = st.number_input("历史回看步长（分钟数）", min_value=60, value=0, step=60, help="设为0则系统自动计算")
            manual_output = st.number_input("预测未来步长（分钟数）", min_value=10, value=0, step=10, help="设为0则系统自动计算")
        with col2:
            epochs = st.number_input("训练迭代轮数", min_value=5, max_value=100, value=25, step=5)
            lr = st.number_input("学习率", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")

    uploaded_file = st.file_uploader("📂 点击上传或拖拽 CSV 文件（需包含「时间」和「负荷」列）", type=["csv"])

    if uploaded_file is not None:
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

        if st.button("🚀 开始智能预测", type="primary", use_container_width=True):
            total_len = len(df)
            input_steps = manual_input if manual_input > 0 else min(720, int(total_len * 0.3))
            output_steps = manual_output if manual_output > 0 else min(60, int(total_len * 0.05))
            if total_len < input_steps + output_steps + 10:
                st.error(f"❌ 数据量不足。当前 {total_len} 条，请减少预测步长。")
                return

            with st.status("⏳ 模型训练中，请稍候...", expanded=True) as status:
                scaler = StandardScaler()
                load_scaled = scaler.fit_transform(df['load'].values.reshape(-1, 1)).flatten()
                X, y = create_sequences(load_scaled, input_steps, output_steps)
                split_idx = int(len(X) * 0.85)
                X_train, y_train = X[:split_idx], y[:split_idx]
                X_val, y_val = X[split_idx:], y[split_idx:]
                
                X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
                y_train_t = torch.FloatTensor(y_train)
                X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
                y_val_t = torch.FloatTensor(y_val)
                
                train_dataset = TensorDataset(X_train_t, y_train_t)
                val_dataset = TensorDataset(X_val_t, y_val_t)
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = TCN(input_size=1, output_size=output_steps, num_channels=[32, 64, 128], kernel_size=5, dropout=0.3)
                
                progress_bar = st.progress(0, text="开始训练...")
                best_state = train_model(model, train_loader, val_loader, epochs, lr, device, progress_bar)
                progress_bar.empty()
                
                model.load_state_dict(best_state)
                model.eval()
                
                last_seq = load_scaled[-input_steps:]
                input_tensor = torch.FloatTensor(last_seq).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_scaled = model(input_tensor).cpu().numpy().flatten()
                
                pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                last_time = df['datetime'].iloc[-1]
                future_times = pd.date_range(start=last_time + pd.Timedelta(minutes=1), periods=output_steps, freq='1T')
                status.update(label="✅ 预测完成！", state="complete")

            st.subheader("🔮 预测结果分析")
            st.markdown(f"""
            <div class="prediction-box">
                <h4 style="margin-top:0;">📈 未来 {output_steps} 分钟预测概览</h4>
                <table style="width:100%;">
                    <tr><td><b>起始预测时刻</b></td><td>{future_times[0].strftime('%Y-%m-%d %H:%M')}</td>
                    <td><b>峰值负荷</b></td><td style="color:#d32f2f; font-weight:bold;">{pred_original.max():.2f}</td></tr>
                    <tr><td><b>结束预测时刻</b></td><td>{future_times[-1].strftime('%Y-%m-%d %H:%M')}</td>
                    <td><b>平均负荷</b></td><td style="color:#1976d2; font-weight:bold;">{pred_original.mean():.2f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            fig_pred, ax_pred = plt.subplots(figsize=(14, 5))
            show_hist = min(240, len(df))
            plot_hist_df = df.iloc[-show_hist:]
            ax_pred.plot(plot_hist_df['datetime'], plot_hist_df['load'], label='历史负荷', linewidth=2, color='#1E88E5')
            ax_pred.plot(future_times, pred_original, label='预测负荷 (AI生成)', linewidth=2.5, color='#FF6F00', marker='o', markersize=5)
            ax_pred.axvline(x=last_time, color='red', linestyle='--', linewidth=1.5, label='当前时刻（预测起点）')
            ax_pred.legend(fontsize=12)
            ax_pred.set_title("用能趋势预测", fontsize=16)
            ax_pred.grid(True, alpha=0.3)
            st.pyplot(fig_pred)

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
