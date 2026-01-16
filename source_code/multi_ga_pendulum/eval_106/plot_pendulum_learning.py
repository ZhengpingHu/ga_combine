# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOG_ROOT_DIR = "./"
OUTPUT_DIR = "summary_analysis_pendulum"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try: plt.style.use('seaborn-v0_8-paper')
except: plt.style.use('seaborn-paper')

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "axes.labelsize": 10, "font.size": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True, "axes.unicode_minus": False
})

COLOR_REWARD = "#1f77b4"
COLOR_FITNESS = "#d62728"

def load_pendulum_runs(root_dir):
    print(f"--- Scanning: {os.path.abspath(root_dir)} ---")
    files = glob.glob(os.path.join(root_dir, "**", "metrics*.csv"), recursive=True)
    if not files:
        print("[Error] No files found."); return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]

            rename_map = {
                'gen': 'generation',
                'best_fitness': 'best_fitness_score',
                'avg_raw_reward': 'global_avg_raw_reward'
            }
            df = df.rename(columns=rename_map)

            if 'generation' in df.columns and 'global_avg_raw_reward' in df.columns:
                dfs.append(df)
            else:
                print(f"[Warn] Skipping {os.path.basename(f)}: Columns mismatch. Found: {df.columns.tolist()}")
        except Exception as e:
            print(f"[Error] {os.path.basename(f)}: {e}")
    
    if not dfs: return None
    print(f"--> Loaded {len(dfs)} runs.")
    return pd.concat(dfs, ignore_index=True)

def process_stats(full_df):
    if full_df is None: return None
    grouped = full_df.groupby("generation")
    stats = grouped.agg({
        'global_avg_raw_reward': ['mean', 'std'],
        'best_fitness_score': ['mean', 'std']
    })
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    return stats.reset_index()

def plot_pendulum_learning(stats):
    if stats is None: return
    fig, ax1 = plt.subplots(figsize=(4.0, 3.2), dpi=300)
    gens = stats['generation']

    mu_rew = stats['global_avg_raw_reward_mean']
    std_rew = stats['global_avg_raw_reward_std'].fillna(0)
    
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Population Avg. Reward', color=COLOR_REWARD, fontweight='bold')
    line1, = ax1.plot(gens, mu_rew, color=COLOR_REWARD, lw=1.5, label='Pop. Avg Reward')
    ax1.fill_between(gens, mu_rew - std_rew, mu_rew + std_rew, color=COLOR_REWARD, alpha=0.15, edgecolor=None)
    ax1.tick_params(axis='y', labelcolor=COLOR_REWARD)
    

    y_min = mu_rew.min()
    ax1.set_ylim(y_min - 100, 50)
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)


    ax2 = ax1.twinx()
    mu_fit = stats['best_fitness_score_mean']
    std_fit = stats['best_fitness_score_std'].fillna(0)
    
    ax2.set_ylabel('Best Fitness Score', color=COLOR_FITNESS, fontweight='bold')
    line2, = ax2.plot(gens, mu_fit, color=COLOR_FITNESS, lw=1.5, ls='--', label='Best Fitness')
    ax2.fill_between(gens, mu_fit - std_fit, mu_fit + std_fit, color=COLOR_FITNESS, alpha=0.1, edgecolor=None)
    ax2.tick_params(axis='y', labelcolor=COLOR_FITNESS)
    
    ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color(COLOR_FITNESS)
    ax2.spines['left'].set_color(COLOR_REWARD); ax2.grid(False)
    
    lines = [line1, line2]; labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, frameon=False)

    out = os.path.join(OUTPUT_DIR, "fig_pendulum_learning.pdf")
    plt.savefig(out, format='pdf', bbox_inches='tight')
    print(f"Saved: {out}")

if __name__ == "__main__":
    df = load_pendulum_runs(LOG_ROOT_DIR)
    plot_pendulum_learning(process_stats(df))