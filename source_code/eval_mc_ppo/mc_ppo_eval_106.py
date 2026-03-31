#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================
# 【终极防崩区】：必须放在所有 import 之前！！！
# ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import traceback
import multiprocessing as mp
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ==========================================
# 1. IEEE Styling & Config
# ==========================================
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
    "xtick.labelsize": 10,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": False,
    "figure.constrained_layout.use": True
})

SEED_PALETTE = sns.color_palette("deep", 5) 

# ==========================================
# [配置区] 模型路径配置 (MountainCar)
# ==========================================
MODELS_CLEAN = {
    101: "./ppo_3m_clean_mc_results_seed_101/final_ppo_model.zip",
    102: "./ppo_3m_clean_mc_results_seed_102/final_ppo_model.zip",
    103: "./ppo_3m_clean_mc_results_seed_103/final_ppo_model.zip",
    104: "./ppo_3m_clean_mc_results_seed_104/final_ppo_model.zip",
    105: "./ppo_3m_clean_mc_results_seed_105/final_ppo_model.zip"
}

MODELS_NOISY = {
    101: "./ppo_3m_noisy_mc_results_seed_101/final_ppo_model.zip",
    102: "./ppo_3m_noisy_mc_results_seed_102/final_ppo_model.zip",
    103: "./ppo_3m_noisy_mc_results_seed_103/final_ppo_model.zip",
    104: "./ppo_3m_noisy_mc_results_seed_104/final_ppo_model.zip",
    105: "./ppo_3m_noisy_mc_results_seed_105/final_ppo_model.zip"
}

TEST_BASE_SEED = 106
TEST_ENV_COUNT = 100
MAX_STEPS = 200

# ==========================================
# 2. 环境 Wrapper
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        self.clutter_texture = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)

    def render(self):
        frame = self.env.render()
        if frame is None: return None
        frame = frame.astype(np.float32)

        if self.add_clutter:
            bg_mask = np.all(frame > 240, axis=-1)
            self.clutter_texture = np.roll(self.clutter_texture, shift=3, axis=0)
            self.clutter_texture = np.roll(self.clutter_texture, shift=7, axis=1)
            h, w = frame.shape[:2]
            
            tex_slice = self.clutter_texture[:h, :w]
            frame[bg_mask] = tex_slice[bg_mask].astype(np.float32)

        if self.gaussian_std > 0:
            noise = np.random.normal(0, self.gaussian_std, frame.shape)
            frame += noise

        return np.clip(frame, 0, 255).astype(np.uint8)

class PixelMountainCar(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.img_size = (84, 84) 
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.uint8
        )

    def observation(self, obs):
        img = self.env.render()
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        return img

def generate_test_seeds(base_seed: int, count: int) -> list:
    rng = np.random.default_rng(base_seed)
    return rng.integers(low=0, high=2**31-1, size=count, dtype=np.int32).tolist()

# ==========================================
# 3. 多进程 Worker 任务函数
# ==========================================
def run_single_episode(args):
    model_path, agent_seed, env_seed, test_type, n_stack = args
    
    try:
        model = PPO.load(model_path, device="cpu")
        
        def make_env():
            base_env = gym.make("MountainCar-v0", render_mode="rgb_array")
            if test_type == "noisy":
                env = VisuallyClutteredWrapper(base_env, gaussian_std=25.0, add_clutter=True)
            else:
                env = VisuallyClutteredWrapper(base_env, gaussian_std=0.0, add_clutter=False)
            return PixelMountainCar(env)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)
        
        vec_env.seed(int(env_seed))
        obs = vec_env.reset()
        
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]
            done = dones[0]
            steps += 1
            
        vec_env.close()
        
        return {
            "Agent_Source_Seed": str(agent_seed),
            "Test_Env_Seed": env_seed,
            "Reward": total_reward
        }
    except Exception as e:
        return {
            "Agent_Source_Seed": str(agent_seed),
            "Test_Env_Seed": env_seed,
            "Reward": -200.0 
        }

# ==========================================
# 4. 主评估逻辑
# ==========================================
def evaluate(train_type, test_type, outdir, n_stack, processes=12):
    print(f"\n{'='*60}")
    print(f"[*] 开始评估: [{train_type.upper()}] 权重 -> [{test_type.upper()}] 环境")
    print(f"{'='*60}")
    
    filename_base = f"ppo_mc_{train_type}_to_{test_type}"
    csv_path = os.path.join(outdir, f"{filename_base}.csv")
    pdf_path = os.path.join(outdir, f"{filename_base}.pdf")
    png_path = os.path.join(outdir, f"{filename_base}.png")
    
    # 【智能断点续传】：如果 CSV 已经存在，直接画图！
    if os.path.exists(csv_path):
        print(f"[*] 检测到已存在的评估数据 {csv_path}，跳过耗时的推理阶段，直接进入绘图！")
        df = pd.read_csv(csv_path)
        # 确保 Seed 被读取为字符串，以便于 Seaborn 正确映射颜色
        df['Agent_Source_Seed'] = df['Agent_Source_Seed'].astype(str)
    else:
        try:
            model_paths = MODELS_CLEAN if train_type == "clean" else MODELS_NOISY
            test_seeds = generate_test_seeds(TEST_BASE_SEED, TEST_ENV_COUNT)
            
            tasks = []
            for agent_seed, model_path in model_paths.items():
                if not os.path.exists(model_path):
                    print(f"[!] 警告: 找不到模型文件 {model_path}，已跳过")
                    continue
                for env_seed in test_seeds:
                    tasks.append((model_path, agent_seed, env_seed, test_type, n_stack))

            if not tasks: 
                print("[!] 本阶段没有检测到有效任务，跳过。")
                return

            all_results = []
            total_tasks = len(tasks)
            
            print(f"[*] 任务分发完毕，启动 {processes} 个工作线程...")
            with mp.Pool(processes=processes) as pool:
                for res in tqdm(pool.imap_unordered(run_single_episode, tasks), total=total_tasks, desc=f"Evaluating"):
                    all_results.append(res)

            df = pd.DataFrame(all_results)
            df.to_csv(csv_path, index=False)
            print(f"\n[*] 评估数据已安全保存至: {csv_path}")
            
        except Exception as e:
            print(f"\n[!!!] 阶段 [{train_type} -> {test_type}] 发生崩溃，已拦截错误！")
            traceback.print_exc()
            return

    # 2. 绘制 IEEE 风格小提琴图
    print(f"[*] 正在生成可视化图表...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    try:
        sns.violinplot(
            data=df, y="Reward", color="#ECEFF1", inner=None, 
            linewidth=0, alpha=0.6, bw_adjust=0.2, ax=ax, cut=0             
        )
    except Exception as e:
        print(f"[!] Violin plot 绘制轻微异常: {e}")

    sns.boxplot(
        data=df, y="Reward", width=0.15,
        boxprops={'facecolor': 'none', 'edgecolor': '#333333', 'linewidth': 1.0},
        whiskerprops={'color': '#333333', 'linewidth': 1.0},
        capprops={'color': '#333333', 'linewidth': 1.0},
        medianprops={'color': '#D32F2F', 'linewidth': 2.0, 'alpha': 1.0},
        showfliers=False, zorder=5, ax=ax
    )

    sns.stripplot(
        data=df_shuffled, y="Reward", hue="Agent_Source_Seed",
        palette=SEED_PALETTE, alpha=0.6, size=4, jitter=0.3,       
        edgecolor='white', linewidth=0.1, zorder=2, ax=ax
    )

    ax.set_ylabel("Total Reward (Test)", fontweight='bold')
    ax.set_xlabel("") 
    ax.set_xticklabels([])

    y_min = df['Reward'].min()
    y_max = df['Reward'].max()
    
    view_min = max(-210, y_min - 10)
    view_max = min(-80, y_max + 10) if y_max < -80 else -80
    ax.set_ylim(view_min, view_max) 
    
    ax.axhline(-80, color='#D32F2F', linestyle='-', linewidth=1.0, alpha=0.4, zorder=1, label='Physical Limit')
    
    # Set tick locator first, then draw the grid
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.grid(True, linestyle='--', color='#E0E0E0', alpha=0.8)
    ax.xaxis.grid(False)

    ax.legend(title="Source Seed", bbox_to_anchor=(0.5, 1.02), loc='lower center', 
              ncol=5, frameon=False, borderaxespad=0, columnspacing=1.0, handletextpad=0.4)

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"[*] 图表已安全保存至: {pdf_path}")


def main():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="eval_data_mc")
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--processes", type=int, default=12, help="并发线程数")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    evaluate("clean", "clean", args.outdir, args.n_stack, args.processes)
    evaluate("clean", "noisy", args.outdir, args.n_stack, args.processes)
    evaluate("noisy", "noisy", args.outdir, args.n_stack, args.processes)
    
    print("\n[+] 全流程多进程极速测试运行完毕！")

if __name__ == "__main__":
    main()