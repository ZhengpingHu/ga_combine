# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOG_ROOT_DIR = "runs_cartpole"
OUTPUT_DIR = "summary_analysis"
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
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "axes.unicode_minus": False
})

COLOR_REWARD = "#1f77b4"
COLOR_FITNESS = "#d62728"

def load_all_runs(root_dir):
    print(f"Scanning for metrics.csv in {root_dir}...")
    files = glob.glob(os.path.join(root_dir, "**", "metrics.csv"), recursive=True)
    if not files: return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception: pass
    return pd.concat(dfs, ignore_index=True) if dfs else None

def generate_summary_table(full_df):
    grouped = full_df.groupby("generation")
    stats_df = grouped.agg({
        'selected_individual_reward': ['mean', 'min', 'max'],
        'best_fitness_score': ['mean', 'min', 'max']
    })
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    return stats_df.reset_index()

def plot_combined_curves(stats_df):

    fig, ax1 = plt.subplots(figsize=(4.0, 3.2), dpi=300)

    gens = stats_df['generation']

    mean_reward = stats_df['selected_individual_reward_mean']
    r_min = stats_df['selected_individual_reward_min']
    r_max = stats_df['selected_individual_reward_max']
    
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Avg. Raw Reward', color=COLOR_REWARD, fontweight='bold')
    
    line1, = ax1.plot(gens, mean_reward, color=COLOR_REWARD, linewidth=1.5, label='Env. Reward')
    ax1.fill_between(gens, r_min, r_max, color=COLOR_REWARD, alpha=0.15, label='Reward Range', edgecolor=None)
    
    ax1.tick_params(axis='y', labelcolor=COLOR_REWARD)

    y_max_val = r_max.max()
    ax1.set_ylim(0, y_max_val * 1.1) 
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    mean_fit = stats_df['best_fitness_score_mean']
    f_min = stats_df['best_fitness_score_min']
    f_max = stats_df['best_fitness_score_max']

    ax2.set_ylabel('Avg. Fitness Score', color=COLOR_FITNESS, fontweight='bold')

    line2, = ax2.plot(gens, mean_fit, color=COLOR_FITNESS, linewidth=1.5, linestyle='--', label='Fitness Score')
    ax2.fill_between(gens, f_min, f_max, color=COLOR_FITNESS, alpha=0.1, label='Fitness Range', edgecolor=None)
    
    ax2.tick_params(axis='y', labelcolor=COLOR_FITNESS)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(COLOR_FITNESS)
    ax2.spines['left'].set_color(COLOR_REWARD)
    ax2.grid(False)
    
    f_max_val = f_max.max()
    ax2.set_ylim(0, f_max_val * 1.1)
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, 
               bbox_to_anchor=(0.5, 1.02), 
               loc='lower center', 
               ncol=2, 
               frameon=False,
               borderaxespad=0)

    out_path_pdf = os.path.join(OUTPUT_DIR, "fig_learning_curve.pdf")
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved: {out_path_pdf}")
if __name__ == "__main__":
    df = load_all_runs(LOG_ROOT_DIR)
    if df is not None:
        stats_df = generate_summary_table(df)
        plot_combined_curves(stats_df)
        plt.show()