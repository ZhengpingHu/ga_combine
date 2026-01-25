import pandas as pd
import numpy as np
import glob

def analyze_pendulum_logs(mode_name, file_pattern):
    # 1. 配置
    # Pendulum 任务中，-1000 通常被认为是“成功Swing-up”的门槛
    # -200 左右是理论最优（稳定倒立）
    success_threshold = -1000.0 
    max_gens = 100

    files = glob.glob(file_pattern)
    if not files:
        print(f"Warning: No files found for {mode_name} with pattern '{file_pattern}'")
        return

    print(f"\n--- Analyzing Pendulum-v1 [{mode_name}] ({len(files)} runs detected) ---")

    generations_to_converge = []
    max_rewards = []
    final_avg_rewards = []
    success_count = 0

    for file in files:
        try:
            df = pd.read_csv(file)
            
            # --- 适配您的 CSV 列名 ---
            # gen, best_fitness, max_raw_reward, avg_raw_reward
            gen_col = 'gen'
            best_reward_col = 'max_raw_reward' # 使用 max_raw_reward 作为最佳个体指标
            avg_reward_col = 'avg_raw_reward'

            # 1. 获取该次运行的历史最高分 (Max Reward)
            run_max_reward = df[best_reward_col].max()
            max_rewards.append(run_max_reward)

            # 2. 获取最后一代的平均分
            final_avg = df[avg_reward_col].iloc[-1]
            final_avg_rewards.append(final_avg)

            # 3. 判断是否成功 (是否Swing up)
            if run_max_reward > success_threshold:
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
    # 请确保您的文件名匹配下面的模式
    # 1. 分析 Visual 版本 (您的 HSR 方法)
    analyze_pendulum_logs("Visual (Ours)", "metrics_*.csv")
    
    print("\n" + "="*50 + "\n")
    
    # 2. 分析 Env 版本 (Ground Truth Baseline)
    analyze_pendulum_logs("Env (Baseline)", "metrics_pendulum_env_*.csv")