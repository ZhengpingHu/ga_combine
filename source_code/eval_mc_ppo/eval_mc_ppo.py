#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
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

# ==========================================
# 2. 环境 Wrapper 
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        # MountainCar 原生分辨率是 600x400 (W=600, H=400)
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
            texture_resized = cv2.resize(self.clutter_texture, (w, h))
            frame[bg_mask] = texture_resized[bg_mask].astype(np.float32)

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

# ==========================================
# 3. 测试逻辑
# ==========================================
def generate_test_seeds(base_seed: int, count: int) -> list:
    rng = np.random.default_rng(base_seed)
    return rng.integers(low=0, high=2**31-1, size=count, dtype=np.int32).tolist()

def evaluate(train_type, test_type, outdir, n_stack):
    print(f"\n[*] 开始评估: 训练于 [{train_type.upper()}] 模型 -> 测试于 [{test_type.upper()}] 环境")
    
    model_paths = MODELS_CLEAN if train_type == "clean" else MODELS_NOISY
    test_seeds = generate_test_seeds(TEST_BASE_SEED, TEST_ENV_COUNT)
    all_results = []

    def make_env():
        base_env = gym.make("MountainCar-v0", render_mode="rgb_array")
        if test_type == "noisy":
            env = VisuallyClutteredWrapper(base_env, gaussian_std=25.0, add_clutter=True)
        else:
            env = VisuallyClutteredWrapper(base_env, gaussian_std=0.0, add_clutter=False)
        return PixelMountainCar(env)

    for agent_seed, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"[!] 警告: 找不到模型文件 {model_path}，已跳过")
            continue
            
        print(f"\n>>> 正在加载并测试 PPO Agent {agent_seed} ...")
        model = PPO.load(model_path, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto")

        for env_seed in tqdm(test_seeds, desc=f"Agent {agent_seed} Eval"):
            vec_env = DummyVecEnv([make_env])
            vec_env = VecFrameStack(vec_env, n_stack=n_stack)
            
            vec_env.seed(int(env_seed))
            obs = vec_env.reset()
            
            done = False
            total_reward = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = vec_env.step(action)
                total_reward += rewards[0]
                done = dones[0]
                
            all_results.append({
                "Agent_Source_Seed": str(agent_seed),
                "Test_Env_Seed": env_seed,
                "Reward": total_reward
            })
            vec_env.close()

    if not all_results: return

    # 1. 保存 CSV 数据
    filename_base = f"ppo_mc_{train_type}_to_{test_type}"
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(outdir, f"{filename_base}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[*] 评估数据已保存至: {csv_path}")

    # 2. 绘制 IEEE 风格小提琴图
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    try:
        sns.violinplot(
            data=df,
            y="Reward",
            color="#ECEFF1",
            inner=None, 
            linewidth=0,
            alpha=0.6,
            bw_adjust=0.2,    
            ax=ax,
            cut=0             
        )
    except Exception as e:
        print(f"Violin plot skipped: {e}")

    sns.boxplot(
        data=df,
        y="Reward",
        width=0.15,
        boxprops={'facecolor': 'none', 'edgecolor': '#333333', 'linewidth': 1.0},
        whiskerprops={'color': '#333333', 'linewidth': 1.0},
        capprops={'color': '#333333', 'linewidth': 1.0},
        medianprops={'color': '#D32F2F', 'linewidth': 2.0, 'alpha': 1.0},
        showfliers=False,
        zorder=5,
        ax=ax
    )

    sns.stripplot(
        data=df_shuffled,
        y="Reward",
        hue="Agent_Source_Seed",
        palette=SEED_PALETTE,
        alpha=0.6,      
        size=4,         
        jitter=0.3,       
        edgecolor='white', 
        linewidth=0.1,  
        zorder=2,       
        ax=ax
    )

    ax.set_ylabel("Total Reward (Test)", fontweight='bold')
    ax.set_xlabel("") 
    ax.set_xticklabels([])

    # MountainCar 得分通常在 -200 到 -80 之间
    y_min = df['Reward'].min()
    y_max = df['Reward'].max()
    
    # 设置底部界限，最差就是 -200，稍微留点余量到 -210
    view_min = max(-210, y_min - 10)
    # 顶部界限，最好大概在 -80 左右
    view_max = min(-80, y_max + 10) if y_max < -80 else -80
    ax.set_ylim(view_min, view_max) 
    
    # 理论物理极限界线 (大约在 -80 左右，表示用最少的步数冲上山顶)
    ax.axhline(-80, color='#D32F2F', linestyle='-', linewidth=1.0, alpha=0.4, zorder=1, label='Physical Limit')
    
    ax.yaxis.grid(True, linestyle='--', major_locator=plt.MultipleLocator(20), color='#E0E0E0', alpha=0.8)
    ax.xaxis.grid(False)

    ax.legend(title="Source Seed", 
              bbox_to_anchor=(0.5, 1.02), 
              loc='lower center', 
              ncol=5, 
              frameon=False, 
              borderaxespad=0,
              columnspacing=1.0,
              handletextpad=0.4)

    png_path = os.path.join(outdir, f"{filename_base}.png")
    pdf_path = os.path.join(outdir, f"{filename_base}.pdf")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] 图表保存至: {pdf_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="eval_data_mc")
    parser.add_argument("--n-stack", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    # 自动执行三种测试组合
    evaluate("clean", "clean", args.outdir, args.n_stack)
    evaluate("clean", "noisy", args.outdir, args.n_stack)
    evaluate("noisy", "noisy", args.outdir, args.n_stack)
    
    print("\n[+] 所有测试评估完毕！")

if __name__ == "__main__":
    main()