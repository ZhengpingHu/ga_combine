import pandas as pd
import numpy as np

def analyze_lunarlander_metrics(csv_path):
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 确保列名没有多余的空格
    df.columns = df.columns.str.strip()
    
    # 1. 计算总体的平均 Reward 和标准差 (Avg Reward (Population))
    overall_mean = df['Reward'].mean()
    overall_std = df['Reward'].std()
    
    # 2. 计算成功率 (Run Success Rate: Reward >= 200)
    # 假设每行是一次独立的 test run
    total_runs = len(df)
    successful_runs = len(df[df['Reward'] >= 200])
    success_rate = (successful_runs / total_runs) * 100
    
    # 3. 计算 Best Agent 的表现 (Max Reward (Best Agent))
    # 逻辑：按 Agent_Source_Seed 分组，算出每个 Agent 在这 100 次测试中的平均分
    # 找到平均分最高的那个 Agent，并报告它的 Mean ± Std，同时给出所有测试里的绝对最高分
    agent_stats = df.groupby('Agent_Source_Seed')['Reward'].agg(['mean', 'std', 'max'])
    best_agent_id = agent_stats['mean'].idxmax()
    best_agent_mean = agent_stats.loc[best_agent_id, 'mean']
    best_agent_std = agent_stats.loc[best_agent_id, 'std']
    absolute_max = df['Reward'].max()
    
    # 打印格式化的结果，方便你直接填入 LaTeX 表格
    print("="*50)
    print("LunarLander-v2 Evaluation Analysis Results")
    print("="*50)
    print(f"Total Evaluation Runs: {total_runs}")
    print(f"Absolute Max Reward: {absolute_max:.2f}")
    print("-" * 50)
    print(">>> Data for LaTeX Table <<<")
    print(f"Avg Reward (Population): {overall_mean:.1f} ± {overall_std:.1f}")
    print(f"Max Reward (Best Agent [{best_agent_id}]): {best_agent_mean:.1f} ± {best_agent_std:.1f}")
    print(f"Run Success Rate: {success_rate:.1f}%")
    print("="*50)

if __name__ == "__main__":
    # 请将这里替换为你的实际文件名
    csv_filename = "evaluation_data_106.csv"
    try:
        analyze_lunarlander_metrics(csv_filename)
    except FileNotFoundError:
        print(f"Error: 找不到文件 '{csv_filename}'，请检查路径。")