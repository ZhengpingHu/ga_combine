# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================
# 1. Global Config & IEEE Styling
# ==========================================
INPUT_CSV = "evaluation_results/evaluation_data_106.csv"
OUTPUT_DIR = "submission_plots"
OUTPUT_FILENAME_BASE = "robustness_final_publication"

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
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": False,
    "figure.constrained_layout.use": True
})

# Palette
SEED_PALETTE = sns.color_palette("deep", 5) 

# ==========================================
# 2. Plotting Logic
# ==========================================
def plot_submission_violin_v4():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Data file not found at {INPUT_CSV}")
        # Dummy data: 98% points are exactly 500, 2% are slightly less
        # This simulates the "Champion" behavior accurately
        print("Using Dummy Data for Demo...")
        np.random.seed(42)
        # 490 points are exactly 500.0
        perfect_scores = np.full(490, 500.0)
        # 10 points have minor failures (e.g. 498, 485)
        minor_fails = np.random.uniform(480, 499.9, 10)
        
        data = {
            'Agent_Source_Seed': np.random.choice(['101', '102', '103', '104', '105'], 500),
            'Reward': np.concatenate([perfect_scores, minor_fails])
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(INPUT_CSV)
    
    df['Agent_Source_Seed'] = df['Agent_Source_Seed'].astype(str)
    
    # Shuffle for strip plot visualization
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # --- A. Violin Plot (Optimized for High Concentration) ---
    # cut=0 is MANDATORY here. It prevents the plot from drawing curves above 500.
    # bw_adjust=0.2 makes the KDE fit very tightly to the data (sharp peak at 500).
    try:
        sns.violinplot(
            data=df,
            y="Reward",
            color="#ECEFF1",
            inner=None, 
            linewidth=0,
            alpha=0.6,
            bw_adjust=0.2,    # Sharp peak for concentrated data
            ax=ax,
            cut=0             # STRICT CUTOFF at max/min data
        )
    except Exception as e:
        # Fallback if variance is strictly 0 (Seaborn might error)
        print(f"Violin plot skipped (likely 0 variance): {e}")

    # --- B. Box Plot ---
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

    # --- C. Strip Plot (The Real Hero) ---
    # Showing the raw points is the best proof of "No Exceptions"
    sns.stripplot(
        data=df_shuffled,
        y="Reward",
        hue="Agent_Source_Seed",
        palette=SEED_PALETTE,
        alpha=0.6,      
        size=4,         
        jitter=0.3,       # Spread points out horizontally to show density
        edgecolor='white', 
        linewidth=0.1,  
        zorder=2,       
        ax=ax
    )

    # --- D. Formatting ---
    ax.set_ylabel("Total Reward (Test)", fontweight='bold')
    
    # REMOVED: X-axis label/title
    ax.set_xlabel("") 
    ax.set_xticklabels([])

    # Dynamic Y-Limits
    # If perfect, show a small window near 500
    y_min = df['Reward'].min()
    y_max = df['Reward'].max()
    
    # Zoom in to the top relevant area. 
    # If all are 500, we show 490-500.5 to show it's high.
    view_min = max(480, y_min - (500-y_min)*0.1 if y_min < 500 else 495)
    
    ax.set_ylim(view_min, 500.2) # Strict limit just above 500

    # Add a visual "Ceiling Line" at 500
    ax.axhline(500, color='#D32F2F', linestyle='-', linewidth=1.0, alpha=0.4, zorder=1, label='Physical Limit')

    ax.yaxis.grid(True, linestyle='--', which='major', color='#E0E0E0', alpha=0.8)
    ax.xaxis.grid(False)

    # Legend Setup
    ax.legend(title="Source Seed", 
              bbox_to_anchor=(0.5, 1.02), 
              loc='lower center', 
              ncol=5, 
              frameon=False, 
              borderaxespad=0,
              columnspacing=1.0,
              handletextpad=0.4)

    # --- E. Save ---
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}.pdf")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Final plots saved to:")
    print(f"  [PDF] {pdf_path}")

if __name__ == "__main__":
    plot_submission_violin_v4()