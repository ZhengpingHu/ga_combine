# -*- coding: utf-8 -*-
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 请确保这两个路径指向你真实存放 PPO 文本日志文件的根目录
CLEAN_DIR = "./"        
NOISY_DIR = "./"        

OUTPUT_DIR = "submission_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_STEPS = 1_000_000
RESOLUTION = 10000  # 每 10000 步取一个点

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
    "PPO_Clean": {
        "dir": CLEAN_DIR,
        "file": "train.log",
        "color": "#E57373", # 浅红
        "title": "(a) PPO E2E Learning Curve (Clean Environment)"
    },
    "PPO_Noisy": {
        "dir": NOISY_DIR,
        "file": "train_noise.log",
        "color": "#B71C1C", # 深红
        "title": "(b) PPO E2E Learning Curve (Noisy Environment)"
    }
}

def parse_log_file(filepath):
    """
    【核心升级】支持提取单文件内的多个 Runs(Seeds)
    当检测到 Step 数值减小时，自动切片为新的 Run。
    """
    runs = []
    current_steps = []
    current_rewards = []
    last_step = -1
    
    # 匹配 "[Progress] Steps: 10000 | ... | Recent 20-Ep Avg Reward: 20.0"
    pattern = re.compile(r'Steps:\s*(\d+).*?Reward:\s*([\d\.-]+)')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                reward = float(match.group(2))
                
                # 时间倒流，意味着开始了下一个 Seed 的训练
                if step < last_step:
                    if current_steps:
                        runs.append(pd.DataFrame({'Timestep': current_steps, 'Reward': current_rewards}))
                    current_steps = []
                    current_rewards = []
                
                current_steps.append(step)
                current_rewards.append(reward)
                last_step = step
                
    # 把最后一个 Run 加进去
    if current_steps:
        runs.append(pd.DataFrame({'Timestep': current_steps, 'Reward': current_rewards}))
        
    return runs

def load_and_interpolate_runs(root_dir, filename):
    """扫描具体的日志文件并插值对齐"""
    print(f"\n--- 正在读取: {filename} ---")
    
    # 【修复路径拼接】直接定位到具体文件，而不是当成文件夹去搜索 *.log
    search_path = os.path.join(root_dir, filename)
    
    if not os.path.exists(search_path):
        print(f"[!] 警告: 未找到匹配的日志文件 -> {search_path}")
        return None, None, None
        
    common_timesteps = np.arange(0, MAX_STEPS + RESOLUTION, RESOLUTION)
    interpolated_rewards = []
    
    try:
        run_dfs = parse_log_file(search_path)
        if not run_dfs:
            print(f"  [!] 数据为空，跳过: {search_path}")
            return None, None, None
            
        print(f"  [+] 成功解析: {filename} (自动检测到 {len(run_dfs)} 个独立的 Seed 训练回合)")
        
        for idx, df in enumerate(run_dfs):
            df = df.sort_values(by='Timestep')
            
            # 线性插值 + 前向填充（处理提前停止）
            interp_rew = np.interp(
                common_timesteps, 
                df['Timestep'].values, 
                df['Reward'].values,
                right=df['Reward'].values[-1] 
            )
            
            # 平滑处理曲线
            smooth_rew = pd.Series(interp_rew).rolling(window=10, min_periods=1).mean().values
            interpolated_rewards.append(smooth_rew)
            
    except Exception as e:
        print(f"  [!] 读取失败: {e}")

    if not interpolated_rewards:
        return None, None, None
        
    reward_matrix = np.array(interpolated_rewards)
    mean_reward = np.mean(reward_matrix, axis=0)
    std_reward = np.std(reward_matrix, axis=0)
    
    return common_timesteps, mean_reward, std_reward

def plot_single_task(task_key, config):
    steps, mean_rew, std_rew = load_and_interpolate_runs(config["dir"], config["file"])
    
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
    
    def format_func(value, tick_number):
        if value >= 1_000_000: return f"{value / 1_000_000:.1f}M"
        elif value >= 1_000: return f"{int(value / 1_000)}K"
        return str(int(value))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # 【强制统一 Y 轴】这是分开对比最关键的一步，保证刻度尺绝对一致！
    ax.set_ylim(-10, 520)
    
    # 500 分物理上限参考线
    ax.axhline(500, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.text(MAX_STEPS * 0.02, 505, 'Maximum Physics Reward (500)', color='gray', fontsize=8, alpha=0.8)

    # 图例配置
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax.set_title(config["title"], fontweight='bold', pad=10)

    # 保存
    out_png = os.path.join(OUTPUT_DIR, f"learning_curve_{task_key.lower()}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"learning_curve_{task_key.lower()}.pdf")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    
    # 【核心】清空画布，防止数据黏连
    plt.close(fig)
    print(f"\n[+] {task_key} 独立图表已保存至: {out_png}")

def main():
    print("=========================================")
    print(" 开始生成 PPO 独立学习曲线")
    print("=========================================")
    for task_key, config in TASKS.items():
        plot_single_task(task_key, config)
    print("=========================================")
    print(" 全部图表处理完毕！")

if __name__ == "__main__":
    main()