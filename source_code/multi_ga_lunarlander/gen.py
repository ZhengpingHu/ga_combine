import pandas as pd
import numpy as np

def calculate_overall_convergence(file_prefix="metrics_", start_idx=101, end_idx=105, window_size=10):
    convergence_gens = []
    
    # Exact column names based on your CSV
    gen_col = 'generation' 
    score_col = 'global_max_raw_reward' 
    
    for i in range(start_idx, end_idx + 1):
        file_name = f"{file_prefix}{i}.csv"
        try:
            df = pd.read_csv(file_name)
            # Remove any accidental leading/trailing spaces in column names
            df.columns = df.columns.str.strip()
            
            # Calculate rolling mean to smooth out the severe variance
            df['Rolling_Max'] = df[score_col].rolling(window=window_size, min_periods=1).mean()
            
            # Find the peak of the smoothed curve and the overall range
            max_rolling_val = df['Rolling_Max'].max()
            min_rolling_val = df['Rolling_Max'].min()
            score_range = max_rolling_val - min_rolling_val
            
            # Define threshold as reaching 95% of the total improvement range
            threshold = max_rolling_val - (0.05 * score_range)
            
            # Find the first generation that crosses this threshold
            converge_gen = df[df['Rolling_Max'] >= threshold][gen_col].iloc[0]
            
            convergence_gens.append(converge_gen)
            print(f"Run {i} converged at: Generation {converge_gen} (Smoothed Peak: {max_rolling_val:.2f})")
            
        except FileNotFoundError:
            print(f"Warning: File {file_name} not found. Skipping...")
        except KeyError as e:
            print(f"Error: Column {e} not found in {file_name}. Check headers.")
            return
            
    if convergence_gens:
        mean_gen = np.mean(convergence_gens)
        std_gen = np.std(convergence_gens)
        print("="*50)
        print(">>> Data for LaTeX Table <<<")
        print(f"Generations to Converge: {mean_gen:.1f} +/- {std_gen:.1f}")
        print("="*50)

if __name__ == "__main__":
    calculate_overall_convergence()