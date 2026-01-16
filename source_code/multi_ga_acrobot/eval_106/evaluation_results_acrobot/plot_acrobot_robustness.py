# -*- coding: utf-8 -*-
# save as: plot_acrobot_robustness.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ================= Config =================
INPUT_CSV = "./acrobot_eval_data_seed106.csv"
OUTPUT_DIR = "submission_plots_acrobot"
OUTPUT_FILENAME_BASE = "acrobot_robustness_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEEE Styling
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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": False,
    "figure.constrained_layout.use": True
})

SEED_PALETTE = sns.color_palette("deep", 5) 

def plot_acrobot_robustness():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Using dummy data.")
        # Acrobot Dummy Data: Clusters around -80 (Good) to -100
        np.random.seed(42)
        data = {
            'Agent_Source_Seed': np.random.choice(['101', '102', '103', '104', '105'], 500),
            'Reward': np.random.normal(-85, 5, 500) 
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(INPUT_CSV)
    
    df['Agent_Source_Seed'] = df['Agent_Source_Seed'].astype(str)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # 1. Violin (Limit the range)
    # Acrobot rewards are negative. Max is 0.
    sns.violinplot(
        data=df, y="Reward", color="#ECEFF1", inner=None, linewidth=0,
        alpha=0.6, bw_adjust=0.5, ax=ax, cut=0
    )

    # 2. Box Plot
    sns.boxplot(
        data=df, y="Reward", width=0.15,
        boxprops={'facecolor': 'none', 'edgecolor': '#333333'},
        whiskerprops={'color': '#333333'},
        capprops={'color': '#333333'},
        medianprops={'color': '#D32F2F', 'linewidth': 2.0},
        showfliers=False, zorder=5, ax=ax
    )

    # 3. Strip Plot
    sns.stripplot(
        data=df_shuffled, y="Reward", hue="Agent_Source_Seed",
        palette=SEED_PALETTE, alpha=0.6, size=3, jitter=0.25,
        edgecolor='white', linewidth=0, zorder=2, ax=ax
    )

    # Formatting
    ax.set_ylabel("Total Reward (Higher is Better)", fontweight='bold')
    ax.set_xlabel("")
    ax.set_xticklabels([])

    # Y-Axis Logic for Acrobot
    # Typically -500 (Fail) to 0 (Perfect). 
    # Zoom in on the performance area (e.g. -150 to 0)
    y_max = df['Reward'].max()
    y_min = df['Reward'].min()

    ax.set_ylim(y_min - 10, y_max + 5) 
    
    # Draw "Perfect" Line at 0 (Though almost impossible to hit 0 exact)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.3, label='Theoretical Max')

    ax.legend(title="Source Seed", bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=5, frameon=False)

    plt.savefig(os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}.pdf"), format='pdf', bbox_inches='tight')
    print("Acrobot Robustness Plot Saved.")

if __name__ == "__main__":
    plot_acrobot_robustness()