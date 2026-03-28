# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================
# 1. IEEE Styling & Config
# ==========================================
OUTPUT_DIR = "submission_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME_BASE = "system_energy_profiling"

try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True
})

# ==========================================
# [透明度配置区] 
# ==========================================
LINE_ALPHA = 0.35      # 折线的主体透明度 
FILL_UNDER_CURVE = True # 是否在曲线下方绘制积分阴影
FILL_ALPHA = 0.1       # 曲线下方阴影的透明度 
LINE_WIDTH = 0.8       # 折线粗细 

# ==========================================
# 2. Data Loading & Settings
# ==========================================
DATA_FILES = {
    "PPO (No Noise)": "./power_log_PPO_no_noise.csv",
    "PPO (Noisy)": "./power_log_PPO_noise.csv",
    "GA+YOLO (No Noise)": "./power_log_ga_no_noise.csv",
    "GA+YOLO (Noisy)": "./power_log_ga_noise.csv"
}

COLORS = {
    "PPO (No Noise)": "#E57373",      # 浅红
    "PPO (Noisy)": "#B71C1C",         # 深红
    "GA+YOLO (No Noise)": "#64B5F6",  # 浅蓝
    "GA+YOLO (Noisy)": "#0D47A1"      # 深蓝
}

LINESTYLES = {
    "PPO (No Noise)": "--",
    "PPO (Noisy)": "-",
    "GA+YOLO (No Noise)": "--",
    "GA+YOLO (Noisy)": "-"
}

SMOOTHING_WINDOW = 10 

GROUPS = {
    "GA": ["GA+YOLO (No Noise)", "GA+YOLO (Noisy)"],
    "PPO": ["PPO (No Noise)", "PPO (Noisy)"]
}

def calculate_total_energy(df, power_col, time_interval=0.5):
    return df[power_col].sum() * time_interval

def plot_group(group_name, keys):
    # 每次调用都显式创建一个全新的画布
    fig, (ax_cpu, ax_gpu) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    
    print(f"\n=== 开始处理并绘制分组: [{group_name}] ===")
    
    valid_lines = 0 # 记录成功画了几条线
    
    for label in keys:
        file_path = DATA_FILES[label]
        if not os.path.exists(file_path):
            print(f"  [!] 警告: 未找到 {label} 的数据文件 -> {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        time_col = df.columns[0]
        cpu_col = df.columns[1]
        gpu_col = df.columns[2]
        
        total_energy_j = calculate_total_energy(df, gpu_col, time_interval=0.5)
        print(f"  [*] {label:<20} | GPU总能耗: {total_energy_j:8.2f} J | CPU平均负载: {df[cpu_col].mean():.1f}%")
        
        df['CPU_Smooth'] = df[cpu_col].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        df['GPU_Smooth'] = df[gpu_col].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        
        # --- 绘制 CPU 曲线 ---
        ax_cpu.plot(
            df[time_col], df['CPU_Smooth'], 
            label=label, color=COLORS[label], linestyle=LINESTYLES[label], 
            linewidth=LINE_WIDTH, alpha=LINE_ALPHA 
        )
        if FILL_UNDER_CURVE:
            ax_cpu.fill_between(
                df[time_col], df['CPU_Smooth'], 0, 
                color=COLORS[label], alpha=FILL_ALPHA, zorder=0
            )
        
        # --- 绘制 GPU 曲线 ---
        ax_gpu.plot(
            df[time_col], df['GPU_Smooth'], 
            label=label, color=COLORS[label], linestyle=LINESTYLES[label], 
            linewidth=LINE_WIDTH, alpha=LINE_ALPHA 
        )
        if FILL_UNDER_CURVE:
            ax_gpu.fill_between(
                df[time_col], df['GPU_Smooth'], 0, 
                color=COLORS[label], alpha=FILL_ALPHA, zorder=0
            )
            
        valid_lines += 1

    if valid_lines == 0:
        print(f"  [!] {group_name} 分组没有任何有效数据，跳过出图！")
        plt.close(fig)
        return

    # --- CPU 格式化 ---
    ax_cpu.set_title(f"(a) {group_name} CPU Utilization Over Time", fontweight='bold', pad=10)
    ax_cpu.set_xlabel("Time (Seconds)")
    ax_cpu.set_ylabel("CPU Usage (%)")
    ax_cpu.set_ylim(0, 105)
    
    # --- GPU 格式化 ---
    ax_gpu.set_title(f"(b) {group_name} GPU Power Consumption Over Time", fontweight='bold', pad=10)
    ax_gpu.set_xlabel("Time (Seconds)")
    ax_gpu.set_ylabel("GPU Power (Watts)")
    ax_gpu.set_ylim(bottom=0)
    
    handles, labels = ax_gpu.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=2, frameon=False)

    # 保存
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}_{group_name.lower()}_ieee.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}_{group_name.lower()}_ieee.pdf")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"  [+] {group_name} 图像成功保存至: {png_path}")
    
    # 【核心修复】：释放画布内存，防止下一张图画在同一张底板上或丢失！
    plt.close(fig)

def main():
    print("=========================================")
    print(" 开始生成双系统独立能耗报告")
    print("=========================================")
    for group_name, keys in GROUPS.items():
        plot_group(group_name, keys)
    print("\n[+] 全部两套图表生成完毕！")

if __name__ == "__main__":
    main()