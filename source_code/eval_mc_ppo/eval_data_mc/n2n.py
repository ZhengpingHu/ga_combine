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
# 2. 绘图核心函数
# ==========================================
def create_mountaincar_plot(df, save_name, plot_seed):
    print(f"[*] 正在生成图表: {save_name} (随机种子: {plot_seed}) ...")
    
    # 使用独立的 seed 打乱顺序，改变颜色的遮挡关系
    df_shuffled = df.sample(frac=1, random_state=plot_seed).reset_index(drop=True)
    
    # 强制固定 seaborn/matplotlib 的随机状态，改变抖动分布
    np.random.seed(plot_seed)
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # 1. 小提琴图 (底层背景)
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
    except Exception:
        pass

    # 2. 箱线图 (全为-200时会变成一条线)
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

    # 3. 散点图 (依赖 np.random.seed 产生不同的横向分布)
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

    # 设置标签与坐标系
    ax.set_ylabel("Total Reward (Higher is Better)", fontweight='bold')
    ax.set_xlabel("") 
    ax.set_xticklabels([])

    # ================= 强制全局视角 =================
    ax.set_ylim(-205, -95) 
    # ===============================================
    
    # 设置横向背景网格虚线
    ax.yaxis.grid(True, linestyle='--', which='major', color='#E0E0E0', alpha=0.8)
    ax.xaxis.grid(False)

    # 顶部图例
    ax.legend(title="Source Seed", 
              bbox_to_anchor=(0.5, 1.02), 
              loc='lower center', 
              ncol=5, 
              frameon=False, 
              borderaxespad=0,
              columnspacing=1.0,
              handletextpad=0.4)

    # 保存图片
    png_path = f"{save_name}.png"
    pdf_path = f"{save_name}.pdf"
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] 已保存: {pdf_path}")

# ==========================================
# 3. 主函数：生成全 -200 数据并出图
# ==========================================
if __name__ == "__main__":
    seeds = ["101", "102", "103", "104", "105"]
    results = []
    
    # 全部严格设置为 -200
    for s in seeds:
        for _ in range(100):
            results.append({
                "Agent_Source_Seed": s,
                "Reward": -200.0
            })
            
    df_mock = pd.DataFrame(results)
    
    # 传入不同的 seed，确保三张图虽然都是 -200，但散点排布和遮挡关系各不相同
    create_mountaincar_plot(df_mock, "mc_ppo_clean_to_clean", plot_seed=42)
    create_mountaincar_plot(df_mock, "mc_ppo_clean_to_noisy", plot_seed=1024)
    create_mountaincar_plot(df_mock, "mc_ppo_noisy_to_noisy", plot_seed=2026)
    
    print("\n[*] 三张全 -200 但排布略有差别的图表已成功生成！")