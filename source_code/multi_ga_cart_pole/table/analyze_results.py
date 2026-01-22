import pandas as pd
import numpy as np
import glob
import os

def analyze_cartpole_logs():
    # 1. 配置：文件名模式 (请确保 csv 文件就在当前目录下)
    task_name = "CartPole-v1"
    file_pattern = "metrics_*.csv"  # 假设文件名是 metrics_cartpole_1.csv 等
    success_threshold = 475.0  # CartPole 的成功/收敛阈值
    max_gens = 20              # 您的实验最大代数

    # 查找文件
    files = glob.glob(file_pattern)
    if not files:
        print(f"Error: No files found matching '{file_pattern}' in current directory.")
        return

    print(f"\n--- Analyzing {task_name} ({len(files)} runs detected) ---")

    # 存储每次运行的结果
    generations_to_converge = []
    max_rewards = []
    final_avg_rewards = []
    success_count = 0

    for file in files:
        try:
            df = pd.read_csv(file)
            
            # --- [关键修改] 适配您的 CSV 列名 ---
            gen_col = 'generation'
            # 用 global_max_raw_reward 作为“最佳个体得分”
            best_reward_col = 'global_max_raw_reward' 
            # 用 global_avg_raw_reward 作为“种群平均分”
            avg_reward_col = 'global_avg_raw_reward'

            # 1. 获取该次运行的历史最高分 (Max Reward Best Agent)
            run_max_reward = df[best_reward_col].max()
            max_rewards.append(run_max_reward)

            # 2. 获取最后一代的平均分 (Avg Reward Population)
            final_avg = df[avg_reward_col].iloc[-1]
            final_avg_rewards.append(final_avg)

            # 3. 判断是否成功 (Success Rate)
            # 如果某次运行的历史最高分超过阈值，视为成功
            if run_max_reward >= success_threshold:
                success_count += 1
                
                # 4. 计算收敛代数 (Generations to Converge)
                # 找到第一次达到阈值的代数
                converged_rows = df[df[best_reward_col] >= success_threshold]
                first_gen = converged_rows.iloc[0][gen_col]
                generations_to_converge.append(first_gen)
            else:
                # 未收敛，记为最大代数 (用于统计惩罚)
                generations_to_converge.append(max_gens)

        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # --- 统计函数 (均值 ± 标准差) ---
    def get_stats(data):
        return f"{np.mean(data):.1f} ± {np.std(data):.1f}"

    # --- 输出 LaTeX 填表格式 ---
    print(f"{'Metric':<25} | {'Value (Mean ± Std)':<20}")
    print("-" * 50)
    print(f"{'Generations to Converge':<25} | {get_stats(generations_to_converge)}")
    print(f"{'Max Reward (Best Agent)':<25} | {get_stats(max_rewards)}")
    print(f"{'Avg Reward (Population)':<25} | {get_stats(final_avg_rewards)}")
    print(f"{'Success Rate':<25} | {(success_count / len(files)) * 100:.0f}\\%")

if __name__ == "__main__":
    analyze_cartpole_logs()