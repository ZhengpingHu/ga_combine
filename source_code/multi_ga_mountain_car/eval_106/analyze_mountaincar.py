import pandas as pd
import numpy as np
import glob
import os

def analyze_mountaincar_performance():
    """
    Reads multiple Mountain Car CSV logs and calculates statistical metrics
    for LaTeX table insertion.
    """
    # --- Configuration ---
    TASK_NAME = "MountainCar-v0"
    FILE_PATTERN = "metrics_*.csv"  # Pattern to match your 5 CSV files
    SUCCESS_THRESHOLD = -145.0      # Standard "solved" threshold for MountainCar
    MAX_GENS = 100                  # Maximum generations in your experiment

    # Find all matching files
    files = glob.glob(FILE_PATTERN)
    
    if not files:
        print(f"Warning: No files found matching '{FILE_PATTERN}'")
        return

    print(f"\n--- Analyzing {TASK_NAME} ({len(files)} runs detected) ---")

    # Storage for metrics across runs
    generations_to_converge = []
    max_rewards = []       # Best agent's score in a run
    final_pop_avg = []     # Final population average reward
    success_count = 0

    for file in files:
        try:
            df = pd.read_csv(file)
            
            # --- Column Mapping ---
            # Using the column names you provided:
            # generation, best_fitness_score, global_max_raw_reward, global_avg_raw_reward
            gen_col = 'generation'
            # Use global_max_raw_reward for the "Best Agent" performance check
            best_col = 'global_max_raw_reward' 
            # Use global_avg_raw_reward for population average
            avg_col = 'global_avg_raw_reward'

            # Ensure columns exist (strip whitespace just in case)
            df.columns = [c.strip() for c in df.columns]

            # --- 1. Metric: Max Reward (Best Agent in this run) ---
            run_max_reward = df[best_col].max()
            max_rewards.append(run_max_reward)

            # --- 2. Metric: Final Population Avg Reward ---
            final_avg = df[avg_col].iloc[-1]
            final_pop_avg.append(final_avg)

            # --- 3. Metric: Success Rate ---
            # Check if this run EVER exceeded the success threshold
            is_successful = run_max_reward >= SUCCESS_THRESHOLD
            if is_successful:
                success_count += 1
            
            # --- 4. Metric: Generations to Converge ---
            # Find the first generation where the threshold was met
            converged_rows = df[df[best_col] >= SUCCESS_THRESHOLD]
            if not converged_rows.empty:
                first_gen = converged_rows.iloc[0][gen_col]
                generations_to_converge.append(first_gen)
            else:
                # If never converged, penalize with max_gens (or max_gens + 1)
                generations_to_converge.append(MAX_GENS)

        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # --- Statistical Calculations (Mean ± Std) ---
    def fmt_stat(data):
        return f"{np.mean(data):.1f} ± {np.std(data):.1f}"

    # Calculate final stats dictionary
    stats = {
        "Generations": fmt_stat(generations_to_converge),
        "Max Reward": fmt_stat(max_rewards),
        "Avg Reward": fmt_stat(final_pop_avg),
        "Success Rate": f"{(success_count / len(files)) * 100:.0f}\%" 
    }

    # --- Output Table for LaTeX ---
    print("\n" + "="*50)
    print(f"{'Metric':<30} | {'Value (Mean ± Std)':<20}")
    print("-" * 50)
    print(f"{'Generations to Converge':<30} | {stats['Generations']}")
    print(f"{'Max Reward (Best Agent)':<30} | {stats['Max Reward']}")
    print(f"{'Avg Reward (Population)':<30} | {stats['Avg Reward']}")
    print(f"{'Success Rate':<30} | {stats['Success Rate']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    analyze_mountaincar_performance()