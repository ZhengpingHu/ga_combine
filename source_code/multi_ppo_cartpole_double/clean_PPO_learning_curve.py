# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 请确保这个路径指向你存放 ppo_3m_noisy_results_seed_101 等文件夹的根目录
ROOT_DIR = "./"        

OUTPUT_DIR = "submission_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 你的 PPO 跑了 100万步
MAX_STEPS = 1_000_000
RESOLUTION = 10000  # 我们之前设置的回调是每 10000 步记录一次

# ================= 绘图风格设置 (IEEE CEC风格) =================
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
    "axes.unicode_minus": False
})

# ================= 任务定义区 =================
# 分别定义两个独立图表的属性
TASKS = {
    "PPO_3M_Clean": {
        "folder_pattern": "ppo_3m_clean_results_seed_*",
        "color": "#E57373", # 浅红
        "title": "(a) PPO 3M Learning Curve (Clean Environment)"
    },
    "PPO_3M_Noisy": {
        "folder_pattern": "ppo_3m_noisy_results_seed_*",
        "color": "#B71C1C", # 深红
        "title": "(b) PPO 3M Learning Curve (Noisy Environment)"
    }
}

def load_and_interpolate_csv_runs(root_dir, folder_pattern):
    """
    【升级版数据读取】扫描符合命名规则的文件夹，提取里面的 ppo_learning_curve.csv
    并进行插值对齐。
    """
    print(f"\n--- 正在扫描文件夹: {folder_pattern} ---")
    
    # 拼接路径，寻找所有匹配文件夹下的 CSV 文件
    search_path = os.path.join(root_dir, folder_pattern, "ppo_learning_curve.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"[!] 警告: 未找到任何匹配的 CSV 文件 -> {search_path}")
        return None, None, None
        
    print(f"--> 找到 {len(files)} 个 CSV 数据源: {[os.path.basename(os.path.dirname(f)) for f in files]}")
    
    common_timesteps = np.arange(0, MAX_STEPS + RESOLUTION, RESOLUTION)
    interpolated_rewards = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty or 'Timestep' not in df.columns or 'Reward' not in df.columns:
                print(f"  [-] 数据无效或为空，跳过: {f}")
                continue
                
            df = df.sort_values(by='Timestep')
            
            # 线性插值 + 前向填充（如果提早停止，分数保持平推）
            interp_rew = np.interp(
                common_timesteps, 
                df['Timestep'].values, 
                df['Reward'].values,
                right=df['Reward'].values[-1] 
            )
            
            # 滑动平均稍微平滑一下曲线
            smooth_rew = pd.Series(interp_rew).rolling(window=5, min_periods=1).mean().values
            interpolated_rewards.append(smooth_rew)
            print(f"  [+] 成功解析: {os.path.basename(os.path.dirname(f))}")
            
        except Exception as e:
            print(f"  [!] 读取失败: {e}")

    if not interpolated_rewards:
        return None, None, None
        
    # 转换为 Numpy 矩阵并计算均值和标准差 (阴影部分)
    reward_matrix = np.array(interpolated_rewards)
    mean_reward = np.mean(reward_matrix, axis=0)
    std_reward = np.std(reward_matrix, axis=0)
    
    return common_timesteps, mean_reward, std_reward

def plot_single_task(task_key, config):
    steps, mean_rew, std_rew = load_and_interpolate_csv_runs(ROOT_DIR, config["folder_pattern"])
    
    if steps is None:
        print(f"[!] {task_key} 数据提取失败，跳过出图。")
        return

    # 创建独立的画布
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=300)
    color = config["color"]
    
    # 绘制均值线与阴影
    ax.plot(steps, mean_rew, color=color, linewidth=1.8, label='Population Avg. Reward')
    ax.fill_between(steps, mean_rew - std_rew, mean_rew + std_rew, 
                    color=color, alpha=0.15, edgecolor=None, label='Reward Std Dev')

    # X, Y 轴格式化
    ax.set_xlabel('Simulation Timesteps')
    ax.set_ylabel('Avg. Episode Reward', fontweight='bold', color=color)
    
    # X 轴 M / K 格式化
    def format_func(value, tick_number):
        if value >= 1_000_000: return f"{value / 1_000_000:.1f}M"
        elif value >= 1_000: return f"{int(value / 1_000)}K"
        return str(int(value))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # 【强制统一 Y 轴】保证刻度尺绝对一致！
    ax.set_ylim(-10, 520)
    
    # 500 分物理上限参考线
    ax.axhline(500, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.text(MAX_STEPS * 0.02, 505, 'Maximum Physics Reward (500)', color='gray', fontsize=8, alpha=0.8)

    # 图例配置
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax.set_title(config["title"], fontweight='bold', pad=10)

    # 保存图表
    out_png = os.path.join(OUTPUT_DIR, f"learning_curve_{task_key.lower()}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"learning_curve_{task_key.lower()}.pdf")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    
    # 【核心】清空画布，防止数据黏连
    plt.close(fig)
    print(f"\n[+] {task_key} 独立图表已保存至: {out_png}")

def main():
    print("=========================================")
    print(" 开始生成 PPO 独立学习曲线 (CSV 读取版)")
    print("=========================================")
    for task_key, config in TASKS.items():
        plot_single_task(task_key, config)
    print("=========================================")
    print(" 全部图表处理完毕！")

if __name__ == "__main__":
    main()