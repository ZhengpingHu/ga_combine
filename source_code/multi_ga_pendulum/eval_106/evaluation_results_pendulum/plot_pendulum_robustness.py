# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ================= Config =================
INPUT_CSV = "./pendulum_eval_seed106.csv"
OUTPUT_DIR = "submission_plots_pendulum"
OUTPUT_FILENAME = "fig_pendulum_robustness_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEEE Styling
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
SEED_PALETTE = sns.color_palette("deep", 5) 

# ================= Plotting =================
def plot_pendulum_robustness():
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] File not found: {INPUT_CSV}")
        return

    print(f"--- Loading data from {INPUT_CSV} ---")
    
    # 1. 智能读取：自动检测分隔符 (sep=None)，且去除列名空格
    try:
        df = pd.read_csv(INPUT_CSV, sep=None, engine='python')
        df.columns = [c.strip() for c in df.columns] # 去除列名首尾空格
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return

    # 2. 列名校验
    required_cols = ['Agent_Source_Seed', 'Reward']
    if not all(col in df.columns for col in required_cols):
        print(f"[Error] Columns missing. Found: {df.columns.tolist()}, Expected: {required_cols}")
        return

    # 3. 数据类型转换 & 统计 (不做任何修改，仅打印供核对)
    df['Agent_Source_Seed'] = df['Agent_Source_Seed'].astype(str)
    
    # 打印统计信息，请务必在控制台核对！
    r_min = df['Reward'].min()
    r_max = df['Reward'].max()
    r_mean = df['Reward'].mean()
    zeros = len(df[df['Reward'] == 0])
    
    print(f"Data Summary:")
    print(f"  - Count: {len(df)}")
    print(f"  - Min Reward: {r_min}")
    print(f"  - Max Reward: {r_max}")
    print(f"  - Mean Reward: {r_mean:.2f}")
    print(f"  - Exact 0.0 values: {zeros}")
    
    if zeros > 0:
        print("[Warning] Dataset contains values exactly equal to 0. Please verify if this is valid.")

    # 4. 绘图 (还原数据本身)
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    
    df_shuff = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # A. Violin: cut=0 确保不无中生有画出数据范围之外的形状
    sns.violinplot(data=df, y="Reward", color="#ECEFF1", inner=None, linewidth=0,
                   alpha=0.6, bw_adjust=0.5, cut=0, ax=ax)
    
    # B. Box Plot
    sns.boxplot(data=df, y="Reward", width=0.15, 
                boxprops={'facecolor':'none', 'edgecolor':'#333333'},
                medianprops={'color':'#D32F2F', 'linewidth':2}, 
                showfliers=False, zorder=5, ax=ax)
    
    # C. Strip Plot: 仅展示点，不修改值
    sns.stripplot(data=df_shuff, y="Reward", hue="Agent_Source_Seed", palette=SEED_PALETTE,
                  alpha=0.6, size=3, jitter=0.25, edgecolor='white', 
                  linewidth=0, zorder=2, ax=ax)

    # Formatting
    ax.set_ylabel("Total Reward (Higher is Better)", fontweight='bold')
    ax.set_xlabel(""); ax.set_xticklabels([])
    
    # Y-Axis: 根据真实数据自动调整，留 10% 边距
    margin = (r_max - r_min) * 0.1 if r_max != r_min else 10
    ax.set_ylim(r_min - margin, r_max + margin)
    
    # 只有当数据真的在 0 附近时才画 0 线，否则画了也没意义
    if r_max > -200:
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Ideal (0)')

    ax.legend(title="Source Seed", bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=5, frameon=False)

    out_pdf = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.pdf")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved: {out_pdf}")

if __name__ == "__main__":
    plot_pendulum_robustness()