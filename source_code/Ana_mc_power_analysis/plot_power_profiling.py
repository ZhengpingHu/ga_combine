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
# [透明度配置区] (严格保留您的原始设置)
# ==========================================
LINE_ALPHA = 0.85      # 折线的主体透明度 (稍微调高一点保证红色蓝色清晰可见，您可依喜好改回0.35)
FILL_UNDER_CURVE = True # 是否在曲线下方绘制积分阴影
FILL_ALPHA = 0.1       # 曲线下方阴影的透明度 
LINE_WIDTH = 1.2       # 折线粗细 

# ==========================================
# 2. Data Loading & Settings
# ==========================================
DATA_FILES = {
    "PPO (No Noise)": "mc_ppo_clean_power_log.csv",
    "PPO (Noisy)": "mc_ppo_noisy_power_log.csv",
    "GA+YOLO (No Noise)": "mc_ga_clean_power_log.csv",
    "GA+YOLO (Noisy)": "mc_ga_noisy_power_log.csv"
}

# 导师要求：PPO用红色，GA用蓝色
COLORS = {
    "PPO (No Noise)": "#d62728",      # 学术红
    "PPO (Noisy)": "#d62728",         # 学术红
    "GA+YOLO (No Noise)": "#1f77b4",  # 学术蓝
    "GA+YOLO (Noisy)": "#1f77b4"      # 学术蓝
}

# 因为分成了不同的图，全都用实线即可，视觉上最清晰
LINESTYLES = {
    "PPO (No Noise)": "-",
    "PPO (Noisy)": "-",
    "GA+YOLO (No Noise)": "-",
    "GA+YOLO (Noisy)": "-"
}

# 优化图例显示文字：去掉冗余的环境描述，直接显示算法名
LEGEND_LABELS = {
    "PPO (No Noise)": "Baseline: E2E PPO",
    "PPO (Noisy)": "Baseline: E2E PPO",
    "GA+YOLO (No Noise)": "Ours: Vision-HSR-GA",
    "GA+YOLO (Noisy)": "Ours: Vision-HSR-GA"
}

SMOOTHING_WINDOW = 10 

# 导师要求：按 Clean 和 Noisy 分组
GROUPS = {
    "Clean (Noise-Free)": ["PPO (No Noise)", "GA+YOLO (No Noise)"],
    "Noisy (Visual Disturbances)": ["PPO (Noisy)", "GA+YOLO (Noisy)"]
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
        
        # 使用映射后的优雅图例名称
        display_label = LEGEND_LABELS[label]
        
        # --- 绘制 CPU 曲线 ---
        ax_cpu.plot(
            df[time_col], df['CPU_Smooth'], 
            label=display_label, color=COLORS[label], linestyle=LINESTYLES[label], 
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
            label=display_label, color=COLORS[label], linestyle=LINESTYLES[label], 
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
    ax_cpu.set_title(f"(a) CPU Utilization", fontweight='bold', pad=10)
    ax_cpu.set_xlabel("Time (Seconds)")
    ax_cpu.set_ylabel("CPU Usage (%)")
    ax_cpu.set_ylim(0, 105)
    
    # --- GPU 格式化 ---
    ax_gpu.set_title(f"(b) GPU Power Consumption", fontweight='bold', pad=10)
    ax_gpu.set_xlabel("Time (Seconds)")
    ax_gpu.set_ylabel("GPU Power (Watts)")
    ax_gpu.set_ylim(bottom=0)
    
    # 整个 Figure 的大标题（表明当前环境）
    fig.suptitle(f"System Profiling: {group_name} Environment", fontweight='bold', fontsize=13)
    
    # 图例设置
    handles, labels = ax_gpu.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=2, frameon=False)

    # 保存
    safe_name = group_name.split()[0].lower() # 提取 clean 或 noisy
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}_{safe_name}_ieee.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}_{safe_name}_ieee.pdf")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"  [+] {group_name} 图像成功保存至: {pdf_path}")
    
    # 释放画布内存
    plt.close(fig)

def main():
    print("=========================================")
    print(" 开始生成按环境分类的双系统独立能耗报告")
    print("=========================================")
    for group_name, keys in GROUPS.items():
        plot_group(group_name, keys)
    print("\n[+] 全部两套图表生成完毕！请查看 submission_plots 文件夹。")

if __name__ == "__main__":
    main()