import pandas as pd
import numpy as np
import glob

def analyze_acrobot_logs():
    # 1. 配置
    task_name = "Acrobot-v1"
    file_pattern = "metrics_*.csv"  # 确保文件名匹配
    success_threshold = -100.0 # Acrobot 成功标准
    max_gens = 100             # Acrobot 最大代数通常设为 100

    files = glob.glob(file_pattern)
    if not files:
        print(f"Error: No files found for {task_name}")
        return

    print(f"\n--- Analyzing {task_name} ({len(files)} runs detected) ---")

    generations_to_converge = []
    max_rewards = []
    final_avg_rewards = []
    success_count = 0

    for file in files:
        try:
            df = pd.read_csv(file)
            
            # 适配您的列名
            gen_col = 'generation'
            best_reward_col = 'global_max_raw_reward'
            avg_reward_col = 'global_avg_raw_reward'

            # 1. 获取该次运行的历史最高分
            run_max_reward = df[best_reward_col].max()
            max_rewards.append(run_max_reward)

            # 2. 获取最后一代的平均分
            final_avg = df[avg_reward_col].iloc[-1]
            final_avg_rewards.append(final_avg)

            # 3. 判断是否成功
            if run_max_reward > success_threshold: # 注意是大于 -100
                success_count += 1
                
                # 4. 计算收敛代数
                converged_rows = df[df[best_reward_col] > success_threshold]
                if not converged_rows.empty:
                    first_gen = converged_rows.iloc[0][gen_col]
                    generations_to_converge.append(first_gen)
                else:
                    generations_to_converge.append(max_gens)
            else:
                generations_to_converge.append(max_gens)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # 统计输出
    def get_stats(data):
        return f"{np.mean(data):.1f} ± {np.std(data):.1f}"

    print(f"{'Metric':<25} | {'Value (Mean ± Std)':<20}")
    print("-" * 50)
    print(f"{'Generations to Converge':<25} | {get_stats(generations_to_converge)}")
    print(f"{'Max Reward (Best Agent)':<25} | {get_stats(max_rewards)}")
    print(f"{'Avg Reward (Population)':<25} | {get_stats(final_avg_rewards)}")
    print(f"{'Success Rate':<25} | {(success_count / len(files)) * 100:.0f}\\%")

if __name__ == "__main__":
    analyze_acrobot_logs()