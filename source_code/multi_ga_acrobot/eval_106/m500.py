#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
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
# 2. 生成全 -500 的数据并绘图
# ==========================================
def generate_and_plot_locked_acrobot():
    print("[*] 正在生成全 -500 分的测试数据...")
    
    # 生成假数据：5个模型，每个模型测100次，得分全是 -500
    seeds = ["101", "102", "103", "104", "105"]
    results = []
    for s in seeds:
        for i in range(100):
            results.append({
                "Agent_Source_Seed": s,
                "Test_Env_Seed": 1000 + i,
                "Reward": -500.0
            })
            
    df = pd.DataFrame(results)
    
    # 打乱顺序，防止不同 seed 的散点完全覆盖
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("[*] 正在绘制锁定视角的图表...")
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # 提琴图 (由于方差为0，KDE可能会失败，所以做异常处理，这不会影响最终效果)
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
        pass

    # 箱线图 (此时会变成压在 -500 的一条线)
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

    # 散点图 (加一点抖动，让500个点在 -500 的水平线上散开)
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

    # ================= 强制全局视角 =================
    ax.set_ylim(-505, 5) 
    # ===============================================
    
    # 理论上限是 0 分
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
    outdir = "."
    png_path = os.path.join(outdir, "ppo_acrobot_all_fail_fixed.png")
    pdf_path = os.path.join(outdir, "ppo_acrobot_all_fail_fixed.pdf")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] 图表生成完毕，已保存至: {pdf_path}")

if __name__ == "__main__":
    generate_and_plot_locked_acrobot()