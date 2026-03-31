#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 1. Global Configuration Area
# ==========================================
ROOT_DIR = "./"        
OUTPUT_DIR = "submission_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation parameters (Acrobot PPO ran for 3M steps, recorded every 10K)
MAX_STEPS = 3_000_000 
RESOLUTION = 10000    

# ==========================================
# 2. Environment Specific Axis Settings (Acrobot)
# ==========================================
# Acrobot standard rewards range from -500 (timeout) to 0 (optimal)
Y_MIN = -520
Y_MAX = 20
TARGET_REWARD = 0
TARGET_LABEL = f"Maximum Physics Reward ({TARGET_REWARD})"

# ==========================================
# 3. Tasks Definition (Separate Plots)
# ==========================================
TASKS = {
    "acrobot_clean": {
        "folder_pattern": "ppo_3m_clean_acrobot_results_seed_*",
        "color": "#2C3E50", # Dark Blue
        "title": "(a) PPO 3M Learning Curve (Acrobot Clean)"
    },
    "acrobot_noisy": {
        "folder_pattern": "ppo_3m_noisy_acrobot_results_seed_*",
        "color": "#E74C3C", # Red
        "title": "(b) PPO 3M Learning Curve (Acrobot Noisy)"
    }
}

# ==========================================
# 4. IEEE CEC Styling Setup
# ==========================================
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "axes.unicode_minus": False # Ensure negative signs render correctly
})

# ==========================================
# 5. Core Processing Functions
# ==========================================
def load_and_interpolate_csv_runs(root_dir, folder_pattern):
    """
    Scans folders matching the pattern, extracts ppo_learning_curve.csv,
    and performs linear interpolation to align timesteps across different runs.
    """
    print(f"\n--- Scanning folders matching: {folder_pattern} ---")
    
    search_path = os.path.join(root_dir, folder_pattern, "ppo_learning_curve.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"[!] Warning: No matching CSV files found for pattern -> {folder_pattern}")
        return None, None, None
        
    print(f"--> Found {len(files)} data sources.")
    
    common_timesteps = np.arange(0, MAX_STEPS + RESOLUTION, RESOLUTION)
    interpolated_rewards = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty or 'Timestep' not in df.columns or 'Reward' not in df.columns:
                print(f"  [-] Invalid or empty data, skipping: {f}")
                continue
                
            df = df.sort_values(by='Timestep')
            
            # Linear interpolation with forward fill for early stops
            interp_rew = np.interp(
                common_timesteps, 
                df['Timestep'].values, 
                df['Reward'].values,
                right=df['Reward'].values[-1] 
            )
            
            # Apply a rolling mean to smooth out severe spikes
            smooth_rew = pd.Series(interp_rew).rolling(window=5, min_periods=1).mean().values
            interpolated_rewards.append(smooth_rew)
            print(f"  [+] Successfully parsed: {os.path.basename(os.path.dirname(f))}")
            
        except Exception as e:
            print(f"  [!] Failed to read: {e}")

    if not interpolated_rewards:
        return None, None, None
        
    # Convert to matrix and calculate mean/std for plotting
    reward_matrix = np.array(interpolated_rewards)
    mean_reward = np.mean(reward_matrix, axis=0)
    std_reward = np.std(reward_matrix, axis=0)
    
    return common_timesteps, mean_reward, std_reward

def plot_single_task(task_key, config):
    """
    Generates a standalone plot for a specific task configuration.
    """
    steps, mean_rew, std_rew = load_and_interpolate_csv_runs(ROOT_DIR, config["folder_pattern"])
    
    if steps is None:
        print(f"[!] Task '{task_key}' data extraction failed. Skipping plot.")
        return

    # Create an independent figure
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=300)
    color = config["color"]
    
    # Plot mean line and standard deviation shaded area
    ax.plot(steps, mean_rew, color=color, linewidth=1.8, label='Population Avg. Reward')
    ax.fill_between(steps, mean_rew - std_rew, mean_rew + std_rew, 
                    color=color, alpha=0.15, edgecolor=None, label='Reward Std Dev')

    # Axis formatting
    ax.set_xlabel('Simulation Timesteps')
    ax.set_ylabel('Avg. Episode Reward', fontweight='bold', color=color)
    
    # Format X-axis to display M (Millions) or K (Thousands)
    def format_func(value, tick_number):
        if value >= 1_000_000: return f"{value / 1_000_000:.1f}M"
        elif value >= 1_000: return f"{int(value / 1_000)}K"
        return str(int(value))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # Apply dynamic Y-axis limits
    ax.set_ylim(Y_MIN, Y_MAX)
    
    # Draw the target reward reference line
    ax.axhline(TARGET_REWARD, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
    
    # Position the text slightly below the line (-25) since Y_MAX is 20 and target is 0
    ax.text(MAX_STEPS * 0.02, TARGET_REWARD - 25, TARGET_LABEL, color='gray', fontsize=8, alpha=0.8)

    # Finalize plot details
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax.set_title(config["title"], fontweight='bold', pad=10)

    # Save outputs
    out_png = os.path.join(OUTPUT_DIR, f"learning_curve_{task_key}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"learning_curve_{task_key}.pdf")
    
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    
    # Core: Close the figure to prevent data overlap in memory
    plt.close(fig)
    print(f"\n[+] '{task_key}' standalone plot successfully saved to: {out_png}")

def main():
    print("=========================================")
    print(" Generating Acrobot Independent Learning Curves")
    print("=========================================")
    
    for task_key, config in TASKS.items():
        plot_single_task(task_key, config)
        
    print("=========================================")
    print(" All Acrobot plots have been generated!  ")

if __name__ == "__main__":
    main()