#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CartPole Evaluation Client: Cross-Validation Script IN CLEAN ENVIRONMENT
# Task: Evaluate 5 PPO Agents (Seeds 101-105) on 100 New Environments (Seed 106)

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
# [配置区] 模型路径配置
# 请务必将这里替换为你【在干净环境里训练出的、能拿满分】的那个 PPO 模型路径！
# ==========================================
MODEL_PATHS = {
    101: "./ppo_results_seed_101/final_ppo_model.zip",
    102: "./ppo_results_seed_102/final_ppo_model.zip",
    103: "./ppo_results_seed_103/final_ppo_model.zip",
    104: "./ppo_results_seed_104/final_ppo_model.zip",
    105: "./ppo_results_seed_105/final_ppo_model.zip"
}

TEST_BASE_SEED = 106
TEST_ENV_COUNT = 100

# ==========================================
# 1. 像素化 Wrapper (原汁原味的干净版本)
# ==========================================
class PixelCartPole(gym.ObservationWrapper):
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
# 2. 测试逻辑
# ==========================================
def generate_test_seeds(base_seed: int, count: int) -> list:
    print(f"[*] Generating {count} test seeds from base seed {base_seed}...")
    rng = np.random.default_rng(base_seed)
    return rng.integers(low=0, high=2**31-1, size=count, dtype=np.int32).tolist()

def make_env():
    # 纯净的原始环境，没有任何 Clutter 和 Gaussian Noise
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    return PixelCartPole(base_env)

def main():
    parser = argparse.ArgumentParser()
    # 修改了默认输出文件夹，防止覆盖
    parser.add_argument("--outdir", default="evaluation_ppo_clean_results")
    parser.add_argument("--n-stack", type=int, default=4, help="必须与 PPO 训练时的帧堆叠数一致(4或5)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    test_seeds = generate_test_seeds(TEST_BASE_SEED, TEST_ENV_COUNT)
    
    all_results = []
    
    print(f"\n[*] 开始评估 PPO 模型在【纯净环境】下的基础泛化能力...")
    print(f"[*] 共 {len(MODEL_PATHS)} 个 Agent, 每个测试 {TEST_ENV_COUNT} 局。")

    for agent_seed, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"[!] 警告: 找不到模型文件 {model_path}，已跳过 Agent {agent_seed}")
            continue
            
        print(f"\n>>> 正在加载并测试 PPO Agent {agent_seed} ...")
        
        try:
            model = PPO.load(model_path, device="cuda")
        except Exception as e:
            print(f"[!] 模型加载失败: {e}")
            continue

        for env_seed in tqdm(test_seeds, desc=f"Agent {agent_seed} Eval"):
            vec_env = DummyVecEnv([make_env])
            vec_env = VecFrameStack(vec_env, n_stack=args.n_stack)
            
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

    # ==========================================
    # 3. 数据保存与可视化
    # ==========================================
    if not all_results:
        print("\n[!] 没有收集到任何结果，请检查模型路径！")
        return

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.outdir, "ppo_evaluation_data_clean_106.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[*] 原始数据已完美保存至: {csv_path}")

    summary = df.groupby("Agent_Source_Seed")["Reward"].agg(['mean', 'std', 'min', 'max'])
    print("\n=== PPO 基础泛化统计结果 (纯净环境) ===")
    print(summary)
    
    print(f"\n[*] 正在生成小提琴图...")
    plt.figure(figsize=(12, 8))
    
    # 使用蓝色系代表 Baseline 纯净环境
    sns.violinplot(
        data=df, 
        x="Agent_Source_Seed", 
        y="Reward", 
        hue="Agent_Source_Seed",
        palette="Blues", 
        inner="quartile", 
        legend=False
    )
    
    sns.stripplot(
        data=df, 
        x="Agent_Source_Seed", 
        y="Reward", 
        color="black", 
        alpha=0.3, 
        size=3,
        jitter=True
    )

    plt.title(f"PPO Agent Baseline Evaluation in CLEAN Environment\n(Tested on 100 Unseen Environments generated from Seed {TEST_BASE_SEED})")
    plt.xlabel("PPO Agent ID (Source Training Seed)")
    plt.ylabel("Total Reward")
    plt.ylim(0, 520) 
    plt.grid(True, axis='y', alpha=0.3)
    
    plot_path = os.path.join(args.outdir, "ppo_agent_comparison_clean_violin.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[*] 小提琴图已保存至: {plot_path}")
    print("\n[*] 纯净版 PPO 评估全部完成！")

if __name__ == "__main__":
    main()