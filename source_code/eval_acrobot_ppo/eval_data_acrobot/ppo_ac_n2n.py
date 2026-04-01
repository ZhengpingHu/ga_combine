#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. IEEE Styling & Config
# ==========================================
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
    "xtick.labelsize": 10,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": False,
    "figure.constrained_layout.use": True
})

SEED_PALETTE = sns.color_palette("deep", 5) 

# ==========================================
# 2. 读取与绘图逻辑
# ==========================================
def plot_acrobot_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"[!] 找不到文件: {csv_path}")
        return

    print(f"[*] 正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 确保 Agent_Source_Seed 是字符串，作为类别处理
    df['Agent_Source_Seed'] = df['Agent_Source_Seed'].astype(str)

    # 打乱顺序，防止不同 seed 的散点相互完全覆盖
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # 尝试绘制小提琴图 (如果全都是-500，提琴图可能画不出来，这里做了容错)
    try:
        sns.violinplot(
            data=df,
            y="Reward",
            color="#ECEFF1",
            inner=None, 
            linewidth=0,
            alpha=0.6,
            bw_adjust=0.2,    
            ax=ax,
            cut=0            
        )
    except Exception as e:
        print(f"Violin plot skipped (likely zero variance): {e}")

    # 箱线图
    sns.boxplot(
        data=df,
        y="Reward",
        width=0.15,
        boxprops={'facecolor': 'none', 'edgecolor': '#333333', 'linewidth': 1.0},
        whiskerprops={'color': '#333333', 'linewidth': 1.0},
        capprops={'color': '#333333', 'linewidth': 1.0},
        medianprops={'color': '#D32F2F', 'linewidth': 2.0, 'alpha': 1.0},
        showfliers=False,
        zorder=5,
        ax=ax
    )

    # 散点图
    sns.stripplot(
        data=df_shuffled,
        y="Reward",
        hue="Agent_Source_Seed",
        palette=SEED_PALETTE,
        alpha=0.7,      
        size=4.5,        
        jitter=0.25,      
        edgecolor='white', 
        linewidth=0.1,  
        zorder=2,       
        ax=ax
    )

    ax.set_ylabel("Total Reward (Test)", fontweight='bold')
    ax.set_xlabel("") 
    ax.set_xticklabels([])

    # ================= 关键修复 =================
    # 强制将 Y 轴视角锁定在 Acrobot 的完整物理域：-500 到 0
    # 我们给底部留一点点 buffer (-505)，防止底部的点被坐标轴切掉一半
    ax.set_ylim(-505, 5) 
    # ==========================================
    
    # 理论上限是 0 分 (红线)
    ax.axhline(0, color='#D32F2F', linestyle='-', linewidth=1.0, alpha=0.4, zorder=1, label='Physical Limit')
    
    ax.yaxis.grid(True, linestyle='--', which='major', color='#E0E0E0', alpha=0.8)
    ax.xaxis.grid(False)

    ax.legend(title="Source Seed", 
              bbox_to_anchor=(0.5, 1.02), 
              loc='lower center', 
              ncol=5, 
              frameon=False, 
              borderaxespad=0,
              columnspacing=1.0,
              handletextpad=0.4)

    # 保存图片
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    outdir = os.path.dirname(csv_path)
    if outdir == '': outdir = '.'
    
    png_path = os.path.join(outdir, f"{base_name}_fixed.png")
    pdf_path = os.path.join(outdir, f"{base_name}_fixed.pdf")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] 图表修复完毕并保存至: {pdf_path}")

if __name__ == "__main__":
    # 直接指定你的 CSV 文件名
    target_csv = "ppo_acrobot_noisy_to_noisy.csv"
    plot_acrobot_csv(target_csv)