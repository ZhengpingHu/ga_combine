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
# [配置区] 模型路径配置
# 重要：这里必须指向你用【无噪环境(Clean)】训练出来的 5 个 PPO 模型！
# ==========================================
MODEL_PATHS = {
    101: "./ppo_3m_clean_results_seed_101/final_ppo_model.zip",
    102: "./ppo_3m_clean_results_seed_102/final_ppo_model.zip",
    103: "./ppo_3m_clean_results_seed_103/final_ppo_model.zip",
    104: "./ppo_3m_clean_results_seed_104/final_ppo_model.zip",
    105: "./ppo_3m_clean_results_seed_105/final_ppo_model.zip"
}

TEST_BASE_SEED = 106
TEST_ENV_COUNT = 100

# ==========================================
# 1. 环境 Wrapper (高保真注入噪声)
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        # CartPole 原生分辨率是 600x400
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

class PixelCartPole(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.img_size = (84, 84) # SB3 CNN 默认推荐 84x84
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
    rng = np.random.default_rng(base_seed)
    return rng.integers(low=0, high=2**31-1, size=count, dtype=np.int32).tolist()

def make_env():
    # 核心：评估时强制开启噪声和干扰 (模拟突发环境变化)
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    noisy_env = VisuallyClutteredWrapper(base_env, gaussian_std=25.0, add_clutter=True)
    return PixelCartPole(noisy_env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="eval_data_cartpole")
    parser.add_argument("--n-stack", type=int, default=4, help="必须与 PPO 训练时的帧堆叠数一致(通常是4)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    test_seeds = generate_test_seeds(TEST_BASE_SEED, TEST_ENV_COUNT)
    all_results = []
    
    print(f"\n[*] 开始跨域测试 (Zero-Shot): PPO CartPole [Clean Models] -> [Noisy Environment] ...")

    for agent_seed, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"[!] 警告: 找不到模型文件 {model_path}，已跳过")
            continue
            
        print(f"\n>>> 正在加载并测试 PPO Agent {agent_seed} ...")
        # 强制使用 CUDA 加速推理
        model = PPO.load(model_path, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto")

        for env_seed in tqdm(test_seeds, desc=f"Agent {agent_seed} Eval"):
            vec_env = DummyVecEnv([make_env])
            vec_env = VecFrameStack(vec_env, n_stack=args.n_stack)
            
            vec_env.seed(int(env_seed))
            obs = vec_env.reset()
            
            done = False
            total_reward = 0.0
            
            # CartPole-v1 最大步数是 500 步
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

    # 保存原始 CSV 数据，这是未来合并神图的基石！
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.outdir, "ppo_cartpole_clean_to_noisy.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[*] 原始评估数据已保存至: {csv_path}")

    # 生成一个临时小提琴图供你快速检查结果
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="Agent_Source_Seed", y="Reward", hue="Agent_Source_Seed", palette="Reds", inner="quartile", legend=False)
    sns.stripplot(data=df, x="Agent_Source_Seed", y="Reward", color="black", alpha=0.3, size=3, jitter=True)

    plt.title("PPO CartPole: Clean Model Evaluated on Noisy Environment (Zero-Shot)")
    plt.xlabel("PPO Agent ID (Clean Training Seed)")
    plt.ylabel("Total Reward")
    plt.ylim(0, 520) # CartPole 满分 500
    plt.grid(True, axis='y', alpha=0.3)
    
    plot_path = os.path.join(args.outdir, "ppo_cartpole_clean_to_noisy_violin.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[*] 临时速览图已保存至: {plot_path}")

if __name__ == "__main__":
    main()