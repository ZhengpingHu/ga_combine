# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置 =================
# 请修改为你的 Env 训练 log 目录
LOG_ROOT_DIR = "./" 
OUTPUT_DIR = "summary_analysis_baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEEE 样式
try: plt.style.use('seaborn-v0_8-paper')
except: plt.style.use('seaborn-paper')
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "axes.labelsize": 10, "font.size": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True, "axes.unicode_minus": False
})

# === Env 基准专属配色 (绿色/紫色) ===
COLOR_REWARD = "#2ca02c" # Forest Green
COLOR_FITNESS = "#9467bd" # Purple

def load_env_runs(root_dir):
    print(f"Scanning: {os.path.abspath(root_dir)}")
    files = glob.glob(os.path.join(root_dir, "**", "metrics*.csv"), recursive=True)
    if not files: print("[Error] No files found."); return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            
            # 列名映射 (兼容你的格式)
            rename_map = {
                'gen': 'generation',
                'best_fitness': 'best_fitness_score',
                'avg_raw_reward': 'global_avg_raw_reward',
                'global_avg_raw_reward': 'global_avg_raw_reward' # 防止已经标准化的
            }
            df = df.rename(columns=rename_map)
            
            if 'generation' in df.columns and 'global_avg_raw_reward' in df.columns:
                dfs.append(df)
        except: pass
        
    return pd.concat(dfs, ignore_index=True) if dfs else None

def plot_env_learning(full_df):
    if full_df is None: return
    
    # 聚合计算
    stats = full_df.groupby("generation").agg({
        'global_avg_raw_reward': ['mean', 'std'],
        'best_fitness_score': ['mean', 'std']
    }).reset_index()
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    
    # 绘图
    fig, ax1 = plt.subplots(figsize=(4.0, 3.2), dpi=300)
    gens = stats['generation_']

    # --- 左轴: Reward (Green) ---
    mu_rew = stats['global_avg_raw_reward_mean']
    std_rew = stats['global_avg_raw_reward_std'].fillna(0)
    
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Pop. Avg. Reward (Env)', color=COLOR_REWARD, fontweight='bold')
    
    l1, = ax1.plot(gens, mu_rew, color=COLOR_REWARD, lw=1.5, label='Env. Reward')
    ax1.fill_between(gens, mu_rew - std_rew, mu_rew + std_rew, color=COLOR_REWARD, alpha=0.15, edgecolor=None)
    ax1.tick_params(axis='y', labelcolor=COLOR_REWARD)
    
    # Y轴范围: Env 通常能到 -200。设置范围展示其优越性。
    # 如果数据很好，我们可以把上限设高一点，或者自动调整
    y_min = mu_rew.min()
    ax1.set_ylim(y_min - 100, 50) 
    
    # 增加一条 -200 的虚线 (理论最优)
    ax1.axhline(-200, color='gray', ls='--', lw=1, alpha=0.5, label='Optimal (-200)')

    # --- 右轴: Fitness (Purple) ---
    ax2 = ax1.twinx()
    mu_fit = stats['best_fitness_score_mean']
    std_fit = stats['best_fitness_score_std'].fillna(0)
    
    ax2.set_ylabel('Best Fitness (Env)', color=COLOR_FITNESS, fontweight='bold')
    l2, = ax2.plot(gens, mu_fit, color=COLOR_FITNESS, lw=1.5, ls='--', label='Env. Fitness')
    ax2.fill_between(gens, mu_fit - std_fit, mu_fit + std_fit, color=COLOR_FITNESS, alpha=0.1, edgecolor=None)
    ax2.tick_params(axis='y', labelcolor=COLOR_FITNESS)
    
    # 轴颜色匹配
    ax2.spines['right'].set_color(COLOR_FITNESS)
    ax2.spines['left'].set_color(COLOR_REWARD)
    ax2.grid(False)

    # Legend
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, frameon=False)

    out = os.path.join(OUTPUT_DIR, "fig_env_learning.pdf")
    plt.savefig(out, format='pdf', bbox_inches='tight')
    print(f"Saved: {out}")

if __name__ == "__main__":
    df = load_env_runs(LOG_ROOT_DIR)
    plot_env_learning(df)