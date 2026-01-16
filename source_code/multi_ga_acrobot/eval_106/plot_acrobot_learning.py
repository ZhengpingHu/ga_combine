# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOG_ROOT_DIR = "./"
OUTPUT_DIR = "summary_analysis_acrobot"
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

def load_acrobot_runs(root_dir):
    print(f"--- Scanning directory: {os.path.abspath(root_dir)} ---")

    pattern = os.path.join(root_dir, "**", "metrics*.csv")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print("[Error] No 'metrics*.csv' files found.")
        return None

    print(f"--> Found {len(files)} files: {[os.path.basename(f) for f in files]}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)

            required_cols = ['generation', 'global_avg_raw_reward', 'best_fitness_score']
            
            df.columns = [c.strip() for c in df.columns]
            
            if all(col in df.columns for col in required_cols):
                subset = df[required_cols]
                dfs.append(subset)
            else:
                print(f"[Warning] Skipping {os.path.basename(f)}: Missing columns. Expected {required_cols}")
                
        except Exception as e:
            print(f"[Error] Failed to read {os.path.basename(f)}: {e}")
    
    if not dfs: 
        print("\n[Fatal] No valid dataframes loaded.")
        return None

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"--> Successfully loaded {len(dfs)} runs. Total rows: {len(full_df)}\n")
    return full_df

def process_stats(full_df):
    if full_df is None: return None
    grouped = full_df.groupby("generation")

    stats = grouped.agg({
        'global_avg_raw_reward': ['mean', 'std'], 
        'best_fitness_score': ['mean', 'std']
    })
    
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    return stats.reset_index()

def plot_acrobot_learning_curve(stats_df):
    if stats_df is None: return

    fig, ax1 = plt.subplots(figsize=(4.0, 3.2), dpi=300)
    gens = stats_df['generation']

    mean_rew = stats_df['global_avg_raw_reward_mean']
    std_rew = stats_df['global_avg_raw_reward_std'].fillna(0)
    
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Population Avg. Reward', color=COLOR_REWARD, fontweight='bold')
    
    line1, = ax1.plot(gens, mean_rew, color=COLOR_REWARD, linewidth=1.5, label='Pop. Avg Reward')
    ax1.fill_between(gens, 
                     mean_rew - std_rew, 
                     mean_rew + std_rew, 
                     color=COLOR_REWARD, alpha=0.15, edgecolor=None, label='Reward Std Dev')
    
    ax1.tick_params(axis='y', labelcolor=COLOR_REWARD)

    y_min = mean_rew.min()
    view_min = y_min - abs(y_min)*0.1
    ax1.set_ylim(view_min, 10) 
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5) 

    ax2 = ax1.twinx()
    mean_fit = stats_df['best_fitness_score_mean']
    std_fit = stats_df['best_fitness_score_std'].fillna(0)

    ax2.set_ylabel('Best Fitness Score', color=COLOR_FITNESS, fontweight='bold')
    
    line2, = ax2.plot(gens, mean_fit, color=COLOR_FITNESS, linewidth=1.5, linestyle='--', label='Best Fitness')
    ax2.fill_between(gens, 
                     mean_fit - std_fit, 
                     mean_fit + std_fit, 
                     color=COLOR_FITNESS, alpha=0.1, edgecolor=None, label='Fitness Std Dev')
    
    ax2.tick_params(axis='y', labelcolor=COLOR_FITNESS)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(COLOR_FITNESS)
    ax2.spines['left'].set_color(COLOR_REWARD)
    ax2.grid(False)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    
    ax1.legend(lines, labels, 
               bbox_to_anchor=(0.5, 1.02), 
               loc='lower center', 
               ncol=2, 
               frameon=False, 
               borderaxespad=0)

    out_pdf = os.path.join(OUTPUT_DIR, "fig_acrobot_learning.pdf")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Success! Plot saved to: {out_pdf}")

if __name__ == "__main__":
    df = load_acrobot_runs(LOG_ROOT_DIR)
    if df is not None:
        stats = process_stats(df)
        plot_acrobot_learning_curve(stats)