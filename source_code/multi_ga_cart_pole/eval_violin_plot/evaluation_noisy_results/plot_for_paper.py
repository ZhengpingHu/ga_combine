# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings

# 忽略 Seaborn 对于全常数数据（方差为0）生成 KDE 时的内部警告
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. Global Config & IEEE Styling
# ==========================================
INPUT_CSV = "./evaluation_data_noisy_106.csv" # 确保路径正确
OUTPUT_DIR = "submission_plots"
OUTPUT_FILENAME_BASE = "robustness_final_shadow_v5"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ==========================================
# 2. Plotting Logic
# ==========================================
def plot_submission_violin_shadow():
    if not os.path.exists(INPUT_CSV):
        print(f"[!] Error: 找不到真实数据文件 {INPUT_CSV}")
        print("[*] 正在生成随机测试数据（包含极端低分 14 分）...")
        np.random.seed(42)
        perfect_scores = np.full(480, 500.0)
        minor_fails = np.random.uniform(480, 499.9, 15)
        extreme_fails = np.random.uniform(10, 50, 5) # 包含极低分
        
        data = {
            'Agent_Source_Seed': np.random.choice(['101', '102', '103', '104', '105'], 500),
            'Reward': np.concatenate([perfect_scores, minor_fails, extreme_fails])
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(INPUT_CSV)
    
    df['Agent_Source_Seed'] = df['Agent_Source_Seed'].astype(str)
    
    # 将 Agent 按顺序排列
    order = sorted(df['Agent_Source_Seed'].unique())
    # 使用深色系调色板，确保阴影和散点颜色对应且高级
    SEED_PALETTE = sns.color_palette("deep", len(order))
    
    # Shuffle for strip plot visualization
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)

    # --- A. 钟形阴影 (Violin Plot) ---
    try:
        sns.violinplot(
            data=df,
            x="Agent_Source_Seed",
            y="Reward",
            order=order,
            palette=SEED_PALETTE, # 给阴影上色
            inner=None,           # 去掉内部多余线条
            linewidth=0,          # 无边框，纯阴影效果
            alpha=0.25,           # 【核心】透明度调低，形成柔和的“雾状阴影”
            bw_adjust=0.4,        # 平滑度，0.4能很好地包裹数据点
            ax=ax,
            cut=0                 # 严禁阴影画到数据范围之外
        )
    except Exception as e:
        print(f"Violin plot skipped: {e}")

    # --- B. 极简箱线图 (可选，增加统计学严谨性) ---
    sns.boxplot(
        data=df,
        x="Agent_Source_Seed",
        y="Reward",
        order=order,
        width=0.1,
        boxprops={'facecolor': 'none', 'edgecolor': '#555555', 'linewidth': 0.8},
        whiskerprops={'color': '#555555', 'linewidth': 0.8},
        capprops={'color': '#555555', 'linewidth': 0.8},
        medianprops={'color': '#D32F2F', 'linewidth': 1.5},
        showfliers=False,
        zorder=3,
        ax=ax
    )

    # --- C. 实体数据点 (Strip Plot) ---
    sns.stripplot(
        data=df_shuffled,
        x="Agent_Source_Seed",
        y="Reward",
        order=order,
        hue="Agent_Source_Seed",
        palette=SEED_PALETTE,
        alpha=0.8,        # 【核心】实体点不透明度调高
        size=3.5,         # 点的大小
        jitter=0.25,      # 横向抖动宽度，完美落在钟形阴影内
        edgecolor='white',# 给点加一层白边，立体感拉满
        linewidth=0.3,  
        zorder=4,         # 画在最上层
        ax=ax,
        legend=False
    )

    # --- D. Formatting ---
    ax.set_ylabel("Total Reward (Tested on Unseen Env)", fontweight='bold')
    ax.set_xlabel("Agent Source Seed", fontweight='bold')

    # 动态 Y 轴缩放逻辑：诚实展示包括 14 分在内的下限
    y_min = df['Reward'].min()
    view_min = 0 if y_min < 50 else max(0, y_min - 20)
    ax.set_ylim(view_min, 520) 

    # 500 分满分参考线
    ax.axhline(500, color='#D32F2F', linestyle='-', linewidth=1.2, alpha=0.5, zorder=1)
    ax.text(-0.4, 503, 'Maximum Physical Reward (500)', color='#D32F2F', fontsize=8, alpha=0.8)

    ax.yaxis.grid(True, linestyle='--', which='major', color='#E0E0E0', alpha=0.8)
    ax.xaxis.grid(False)

    # --- E. Save ---
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}.pdf")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Final plots saved to:")
    print(f"  [PDF] {pdf_path}")
    print(f"  [PNG] {png_path}")

if __name__ == "__main__":
    plot_submission_violin_shadow()