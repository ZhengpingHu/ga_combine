#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LunarLander Learning Curve Aggregator (IEEE CEC Edition)
# Features:
# - Red Line/Wave: Best Fitness Score (Mean +/- Std Dev across runs)
# - Blue Line/Wave: Selected Individual Reward (Mean +/- Std Dev across runs)
# - Swapped Z-order: Rewards (Blue) are on top of Fitness (Red)

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= Configuration =================
LOG_ROOT_DIR = "./"  
OUTPUT_DIR = "analysis_plots_lunarlander" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= IEEE Styling =================
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

COLOR_REWARD = "#1f77b4"  # Blue: Reward metrics
COLOR_FITNESS = "#d62728" # Red: Fitness metrics

def load_lunarlander_runs(root_dir):
    """
    Scans the directory for metrics*.csv files and concatenates them.
    """
    print(f"--- Scanning directory: {os.path.abspath(root_dir)} ---")

    pattern = os.path.join(root_dir, "**", "metrics_*.csv")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print("[Error] No 'metrics*.csv' files found.")
        return None

    print(f"--> Found {len(files)} files: {[os.path.basename(f) for f in files]}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            
            # Updated required columns: only focusing on best individual and fitness
            required_cols = [
                'generation', 
                'selected_individual_reward', 
                'best_fitness_score'
            ]
            
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
    print(f"--> Successfully loaded data from {len(dfs)} runs. Total rows: {len(full_df)}\n")
    return full_df

def process_stats(full_df):
    """
    Groups by generation to calculate mean and standard deviation across runs.
    """
    if full_df is None: return None
    grouped = full_df.groupby("generation")

    stats = grouped.agg({
        'selected_individual_reward': ['mean', 'std'],
        'best_fitness_score': ['mean', 'std']
    })
    
    # Flatten multi-level column names 
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    return stats.reset_index()

def plot_lunarlander_learning_curve(stats_df):
    if stats_df is None: return

    # Create the base figure and ax1 (bottom z-order for Red)
    fig, ax1 = plt.subplots(figsize=(4.0, 3.2), dpi=300)
    gens = stats_df['generation']

    # =========================================================================
    # --- Ax1 (Left & Bottom Z-Order): Best Fitness Score (RED) ---
    # =========================================================================
    mean_fit = stats_df['best_fitness_score_mean']
    std_fit = stats_df['best_fitness_score_std'].fillna(0)

    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Best Fitness Score', color=COLOR_FITNESS, fontweight='bold')
    
    line_fit, = ax1.plot(gens, mean_fit, color=COLOR_FITNESS, linewidth=1.5, linestyle='--', label='Best Fitness')
    ax1.fill_between(gens, 
                     mean_fit - std_fit, 
                     mean_fit + std_fit, 
                     color=COLOR_FITNESS, alpha=0.1, edgecolor='none')
    
    ax1.tick_params(axis='y', labelcolor=COLOR_FITNESS)
    
    y_min_fit = (mean_fit - std_fit).min()
    y_max_fit = (mean_fit + std_fit).max()
    padding_fit = abs(y_max_fit - y_min_fit) * 0.1
    ax1.set_ylim(y_min_fit - padding_fit, y_max_fit + padding_fit)
    
    ax1.spines['left'].set_visible(True)
    ax1.spines['left'].set_color(COLOR_FITNESS)

    # =========================================================================
    # --- Ax2 (Right & Top Z-Order): Selected Best Reward (BLUE) ---
    # =========================================================================
    ax2 = ax1.twinx()
    
    mean_rew = stats_df['selected_individual_reward_mean']
    std_rew = stats_df['selected_individual_reward_std'].fillna(0)

    ax2.set_ylabel('Best Reward', color=COLOR_REWARD, fontweight='bold')
    
    line_rew, = ax2.plot(gens, mean_rew, color=COLOR_REWARD, linewidth=1.5, label='Best Reward')
    ax2.fill_between(gens, 
                     mean_rew - std_rew, 
                     mean_rew + std_rew, 
                     color=COLOR_REWARD, alpha=0.15, edgecolor='none')
    
    ax2.tick_params(axis='y', labelcolor=COLOR_REWARD)

    y_min_rew = (mean_rew - std_rew).min()
    y_max_rew = (mean_rew + std_rew).max()
    padding_rew = abs(y_max_rew - y_min_rew) * 0.1
    ax2.set_ylim(y_min_rew - padding_rew, max(250, y_max_rew + padding_rew)) 
    
    # Solved threshold (200) for LunarLander
    ax2.axhline(200, color='gray', linestyle=':', linewidth=1.5, alpha=0.7) 

    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(COLOR_REWARD)
    ax2.grid(False) 

    # =========================================================================
    # --- Legend Aggregation ---
    # =========================================================================
    handles = [line_rew, line_fit]
    labels = ['Best Reward', 'Best Fitness']
    
    ax1.legend(handles, labels, 
               bbox_to_anchor=(0.5, 1.02), 
               loc='lower center', 
               ncol=2, 
               frameon=False, 
               borderaxespad=0)

    # =========================================================================
    # --- Save Output ---
    # =========================================================================
    out_pdf = os.path.join(OUTPUT_DIR, "fig_lunarlander_learning_curve_v2.pdf")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Success! Plot saved to: {out_pdf}")

if __name__ == "__main__":
    df = load_lunarlander_runs(LOG_ROOT_DIR)
    if df is not None:
        stats = process_stats(df)
        plot_lunarlander_learning_curve(stats)