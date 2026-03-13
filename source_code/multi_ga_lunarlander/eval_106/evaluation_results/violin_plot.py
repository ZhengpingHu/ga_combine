# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================
# 1. Global Config & IEEE Styling
# ==========================================
INPUT_CSV = "./evaluation_data_106.csv"
OUTPUT_DIR = "submission_plots"
OUTPUT_FILENAME_BASE = "robustness_final_publication"

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

SEED_PALETTE = sns.color_palette("deep", 5) 

# ==========================================
# 2. Plotting Logic
# ==========================================
def plot_submission_violin_v4():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Data file not found at {INPUT_CSV}")
        print("Using Dummy Data for Demo...")
        np.random.seed(42)
        
        high_scores = np.random.normal(260, 15, 450)
        low_scores = np.random.uniform(50, 200, 50)
        
        data = {
            'Agent_Source_Seed': np.random.choice(['101', '102', '103', '104', '105'], 500),
            'Reward': np.concatenate([high_scores, low_scores])
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(INPUT_CSV)
    
    df['Agent_Source_Seed'] = df['Agent_Source_Seed'].astype(str)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

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
        print(f"Violin plot skipped: {e}")

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

    sns.stripplot(
        data=df_shuffled,
        y="Reward",
        hue="Agent_Source_Seed",
        palette=SEED_PALETTE,
        alpha=0.6,      
        size=4,         
        jitter=0.3,       
        edgecolor='white', 
        linewidth=0.1,  
        zorder=2,       
        ax=ax
    )

    ax.set_ylabel("Total Reward (Test)", fontweight='bold')
    
    ax.set_xlabel("") 
    ax.set_xticklabels([])

    y_min = df['Reward'].min()
    y_max = df['Reward'].max()
    
    margin = (y_max - y_min) * 0.05 if y_max != y_min else 50
    view_min = y_min - margin
    view_max = y_max + margin
    ax.set_ylim(view_min, view_max)

    ax.axhline(200, color='#D32F2F', linestyle='-', linewidth=1.0, alpha=0.4, zorder=1, label='Solved Threshold')

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

    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}.pdf")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Final plots saved to:")
    print(f"  [PDF] {pdf_path}")

if __name__ == "__main__":
    plot_submission_violin_v4()