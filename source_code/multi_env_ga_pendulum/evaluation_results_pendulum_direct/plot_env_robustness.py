# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ================= 配置 =================
INPUT_CSV = "./pendulum_direct_eval_seed106.csv" # 请确认路径
OUTPUT_DIR = "summary_analysis_baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try: plt.style.use('seaborn-v0_8-paper')
except: plt.style.use('seaborn-paper')
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "axes.labelsize": 10, "font.size": 10, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, 
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": False,
    "figure.constrained_layout.use": True
})

# === Env 专属配色 (绿色系) ===
# 使用 'Greens' 色板，取色深一点的段
PALETTE_ENV = sns.color_palette("Greens", 6)[1:] 

def plot_env_robustness():
    if not os.path.exists(INPUT_CSV): 
        print(f"Error: {INPUT_CSV} not found"); return
        
    df = pd.read_csv(INPUT_CSV, sep=None, engine='python')
    df.columns = [c.strip() for c in df.columns]
    
    # 过滤异常值 (同前)
    df = df[df['Reward'] <= -50]
    
    df['Agent_Source_Seed'] = df['Agent_Source_Seed'].astype(str)
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    
    # A. Violin (Green)
    sns.violinplot(data=df, y="Reward", palette="Greens", inner=None, linewidth=0,
                   alpha=0.5, bw_adjust=0.5, cut=0, ax=ax)
    
    # B. Box (Dark Green lines)
    sns.boxplot(data=df, y="Reward", width=0.15, 
                boxprops={'facecolor':'none', 'edgecolor':'#006400'}, # Dark Green
                medianprops={'color':'#800080', 'linewidth':2}, # Purple Median
                showfliers=False, zorder=5, ax=ax)
    
    # C. Strip (Points)
    sns.stripplot(data=df, y="Reward", hue="Agent_Source_Seed", palette=PALETTE_ENV,
                  alpha=0.6, size=3, jitter=0.25, edgecolor='white', 
                  linewidth=0, zorder=2, ax=ax)

    ax.set_ylabel("Total Reward (Env Baseline)", fontweight='bold')
    ax.set_xlabel("")
    ax.set_xticklabels([])
    
    # 范围调整: Env 表现很好，可能在 -400 到 -100 之间
    y_max = df['Reward'].max()
    y_min = df['Reward'].min()
    ax.set_ylim(y_min - 50, y_max + 50)
    
    # -200 参考线
    ax.axhline(-200, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Optimal (-200)')

    ax.legend(title="Env Seed", bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=5, frameon=False)

    out = os.path.join(OUTPUT_DIR, "fig_env_robustness.pdf")
    plt.savefig(out, format='pdf', bbox_inches='tight')
    print(f"Saved: {out}")

if __name__ == "__main__":
    plot_env_robustness()