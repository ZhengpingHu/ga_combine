# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
LOG_ROOT_DIR = "./"  # CSV文件所在的根目录
OUTPUT_DIR = "summary_analysis_mountaincar" # 输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 绘图风格设置 (IEEE CEC风格) =================
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "axes.unicode_minus": False
})

# 定义双轴颜色
COLOR_REWARD = "#1f77b4" # 蓝色：平均奖励
COLOR_FITNESS = "#d62728" # 红色：最佳适应度

def load_mountaincar_runs(root_dir):
    """
    扫描目录，读取所有 metrics*.csv 文件并合并
    """
    print(f"--- Scanning directory: {os.path.abspath(root_dir)} ---")

    # 匹配模式：可以是 metrics_101.csv, metrics-run1.csv 等
    pattern = os.path.join(root_dir, "**", "metrics*.csv")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print("[Error] No 'metrics*.csv' files found.")
        return None

    print(f"--> Found {len(files)} files: {[os.path.basename(f) for f in files]}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)

            # Mountain Car 特定的关键列
            # 蓝色线数据源: global_avg_raw_reward
            # 红色线数据源: best_fitness_score
            required_cols = ['generation', 'global_avg_raw_reward', 'best_fitness_score']
            
            # 去除列名可能存在的空格
            df.columns = [c.strip() for c in df.columns]
            
            if all(col in df.columns for col in required_cols):
                subset = df[required_cols]
                dfs.append(subset)
            else:
                print(f"[Warning] Skipping {os.path.basename(f)}: Missing columns. Expected {required_cols}")
                
        except Exception as e:
            print(f"[Error] Failed to read {os.path.basename(f)}: {e}")
    
    if not dfs: 
        print("\n[Fatal] No valid dataframes loaded.")
        return None

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"--> Successfully loaded {len(dfs)} runs. Total rows: {len(full_df)}\n")
    return full_df

def process_stats(full_df):
    """
    按代数分组，计算均值和标准差
    """
    if full_df is None: return None
    grouped = full_df.groupby("generation")

    stats = grouped.agg({
        'global_avg_raw_reward': ['mean', 'std'], 
        'best_fitness_score': ['mean', 'std']
    })
    
    # 扁平化列名 (e.g., 'global_avg_raw_reward_mean')
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    return stats.reset_index()

def plot_mountaincar_learning_curve(stats_df):
    if stats_df is None: return

    # 创建画布
    fig, ax1 = plt.subplots(figsize=(4.0, 3.2), dpi=300)
    gens = stats_df['generation']

    # --- 左轴 (蓝色): Population Avg Reward ---
    mean_rew = stats_df['global_avg_raw_reward_mean']
    std_rew = stats_df['global_avg_raw_reward_std'].fillna(0)
    
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Population Avg. Reward', color=COLOR_REWARD, fontweight='bold')
    
    line1, = ax1.plot(gens, mean_rew, color=COLOR_REWARD, linewidth=1.5, label='Pop. Avg Reward')
    ax1.fill_between(gens, 
                     mean_rew - std_rew, 
                     mean_rew + std_rew, 
                     color=COLOR_REWARD, alpha=0.15, edgecolor=None, label='Reward Std Dev')
    
    ax1.tick_params(axis='y', labelcolor=COLOR_REWARD)

    # Mountain Car 奖励范围调整 (通常是 -200 到 0)
    # 为了美观，下限留一点空间，上限设为 10 或 0 以容纳标准差波动
    y_min = mean_rew.min()
    view_min = y_min - abs(y_min) * 0.05 # 动态下限
    ax1.set_ylim(view_min, 10)         # 上限设为10，保证0刻度线清晰
    
    # 添加 0 刻度参考线 (理论最大值)
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5) 

    # --- 右轴 (红色): Best Fitness Score ---
    ax2 = ax1.twinx()
    mean_fit = stats_df['best_fitness_score_mean']
    std_fit = stats_df['best_fitness_score_std'].fillna(0)

    ax2.set_ylabel('Best Fitness Score', color=COLOR_FITNESS, fontweight='bold')
    
    line2, = ax2.plot(gens, mean_fit, color=COLOR_FITNESS, linewidth=1.5, linestyle='--', label='Best Fitness')
    ax2.fill_between(gens, 
                     mean_fit - std_fit, 
                     mean_fit + std_fit, 
                     color=COLOR_FITNESS, alpha=0.1, edgecolor=None, label='Fitness Std Dev')
    
    ax2.tick_params(axis='y', labelcolor=COLOR_FITNESS)
    
    # 设置脊柱颜色以匹配数据
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(COLOR_FITNESS)
    ax2.spines['left'].set_color(COLOR_REWARD)
    ax2.grid(False) # 只需要左轴的网格

    # --- 图例合并 ---
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    
    # 图例置于上方外部，避免遮挡曲线
    ax1.legend(lines, labels, 
               bbox_to_anchor=(0.5, 1.02), 
               loc='lower center', 
               ncol=2, 
               frameon=False, 
               borderaxespad=0)

    # 保存
    out_pdf = os.path.join(OUTPUT_DIR, "fig_mountaincar_learning.pdf")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Success! Plot saved to: {out_pdf}")

if __name__ == "__main__":
    # 1. 加载数据
    df = load_mountaincar_runs(LOG_ROOT_DIR)
    
    # 2. 如果数据有效，进行处理和绘图
    if df is not None:
        stats = process_stats(df)
        plot_mountaincar_learning_curve(stats)