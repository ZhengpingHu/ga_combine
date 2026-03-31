#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gymnasium as gym
import numpy as np
import cv2
import time
import math
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from gymnasium import spaces

# ==========================================
# 1. Pixel Wrapper + Energy Shaping (CLEAN)
# ==========================================
class CleanPixelMountainCar(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.img_size = (84, 84)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.uint8
        )
        self.max_energy = -999.0
        self.raw_ep_reward = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.max_energy = -999.0
        self.raw_ep_reward = 0.0
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.raw_ep_reward += reward
        
        # 物理能量提取
        pos, vel = self.env.unwrapped.state
        height = math.sin(3 * pos) * 0.45 + 0.55
        kinetic = (vel ** 2) * 200.0
        current_energy = height + kinetic
        
        shaping_reward = 0.0
        if current_energy > self.max_energy:
            if self.max_energy != -999.0:
                shaping_reward = (current_energy - self.max_energy) * 10.0
            self.max_energy = current_energy
            
        total_train_reward = reward + shaping_reward
        
        if done or truncated:
            info['real_raw_reward'] = self.raw_ep_reward
            
        return self._get_obs(), total_train_reward, done, truncated, info

    def _get_obs(self):
        frame = self.env.render()
        if frame is None:
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        frame = frame.astype(np.float32)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)

# ==========================================
# 2. Strict 10K-Step CSV Logger Callback (Logs RAW REWARD)
# ==========================================
class Strict10kLoggerCallback(BaseCallback):
    def __init__(self, save_path, target_reward=-110.0, check_freq=10000, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.target_reward = target_reward
        self.check_freq = check_freq
        self.last_print_step = 0
        self.start_time = None
        self.save_path = save_path
        self.history_data = [] 

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "real_raw_reward" in info:
                self.episode_rewards.append(info["real_raw_reward"])
        
        if self.num_timesteps - self.last_print_step >= self.check_freq:
            recent_avg = np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) > 0 else -200.0
            elapsed_time = time.time() - self.start_time
            print(f"    [Progress] Steps: {self.num_timesteps} | Elapsed: {elapsed_time:.1f}s | Recent Avg RAW Reward: {recent_avg:.1f}")
            
            self.history_data.append({'Timestep': self.num_timesteps, 'Reward': recent_avg})
            pd.DataFrame(self.history_data).to_csv(self.save_path, index=False)
            self.last_print_step = self.num_timesteps
            
            if len(self.episode_rewards) >= 20 and recent_avg >= self.target_reward:
                print(f"\n    [SUCCESS] Target RAW reward {self.target_reward} reached at step {self.num_timesteps}!")
                return False  
        return True

# ==========================================
# 3. Main Experiment Execution
# ==========================================
if __name__ == "__main__":
    test_seeds = [101, 102, 103, 104, 105]
    num_envs_per_run = 16  
    
    max_training_steps = 3_000_000 
    target_score = -110.0 
    
    print("=== Starting E2E PPO (3M Params) Tests on MOUNTAINCAR IN CLEAN ENVIRONMENT ===")
    
    for run_seed in test_seeds:
        print(f"\n--- Running CLEAN MountainCar Seed {run_seed} ---")
        seed_dir = f"ppo_3m_clean_mc_results_seed_{run_seed}"
        os.makedirs(seed_dir, exist_ok=True)
        csv_save_path = os.path.join(seed_dir, "ppo_learning_curve.csv")
        model_save_path = os.path.join(seed_dir, "final_ppo_model.zip")
        
        vec_env = make_vec_env(
            env_id="MountainCar-v0", n_envs=num_envs_per_run, seed=run_seed,
            vec_env_cls=SubprocVecEnv, wrapper_class=CleanPixelMountainCar,
            env_kwargs={"render_mode": "rgb_array"}
        )
        vec_env = VecFrameStack(vec_env, n_stack=4)
        
        custom_policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=768), net_arch=dict(pi=[256, 256], vf=[256, 256]))
        
        model = PPO("CnnPolicy", vec_env, policy_kwargs=custom_policy_kwargs, verbose=0, device="cuda", seed=run_seed,
                    n_steps=512, batch_size=256, learning_rate=3e-4)
        
        total_params = sum(p.numel() for p in model.policy.parameters())
        print(f"[*] PPO 3M Model Instantiated (Total Params: {total_params:,})")

        robust_cb = Strict10kLoggerCallback(save_path=csv_save_path, target_reward=target_score, check_freq=10000)
        model.learn(total_timesteps=max_training_steps, callback=robust_cb, progress_bar=False)
        
        pd.DataFrame(robust_cb.history_data).to_csv(csv_save_path, index=False)
        model.save(model_save_path)
        vec_env.close()