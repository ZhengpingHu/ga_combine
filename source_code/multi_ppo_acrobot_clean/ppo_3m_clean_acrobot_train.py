#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gymnasium as gym
import numpy as np
import cv2
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from gymnasium import spaces

# ==========================================
# 1. Pixel Wrapper for End-to-End Vision (Clean Environment)
# ==========================================
class PixelAcrobot(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Standard input size for Nature CNN (84x84)
        self.img_size = (84, 84)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.uint8
        )

    def observation(self, obs):
        # Render the raw RGB frame
        img = self.env.render()
        # Resize to match CNN input requirements
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        return img

# ==========================================
# 2. Strict 10K-Step CSV Logger Callback
# ==========================================
class Strict10kLoggerCallback(BaseCallback):
    def __init__(self, save_path, target_reward=-85.0, check_freq=10000, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.target_reward = target_reward
        self.check_freq = check_freq
        self.last_print_step = 0
        self.start_time = None
        self.save_path = save_path
        # List to store the strictly 10k formatted data
        self.history_data = [] 

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Extract reward info if an episode just finished
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        
        # Trigger exactly when passing the next 10,000 step milestone
        if self.num_timesteps - self.last_print_step >= self.check_freq:
            # Calculate rolling average of recent 20 episodes
            recent_avg = np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) > 0 else -500.0
            elapsed_time = time.time() - self.start_time
            
            print(f"    [Progress] Steps: {self.num_timesteps} | Elapsed: {elapsed_time:.1f}s | Recent Avg Reward: {recent_avg:.1f}")
            
            # Append to history and save to CSV
            self.history_data.append({
                'Timestep': self.num_timesteps,
                'Reward': recent_avg
            })
            df = pd.DataFrame(self.history_data)
            df.to_csv(self.save_path, index=False)
            
            self.last_print_step = self.num_timesteps
            
            # Early Stopping Check
            if len(self.episode_rewards) >= 20 and recent_avg >= self.target_reward:
                print(f"\n    [SUCCESS] Target reward {self.target_reward} reached at step {self.num_timesteps}!")
                print(f"    [SUCCESS] Time taken to converge: {elapsed_time:.2f} seconds.")
                return False  
            
        return True

# ==========================================
# 3. Main Experiment Execution
# ==========================================
if __name__ == "__main__":
    # Full batch of seeds for rigorous ablation
    test_seeds = [102, 103, 104, 105]
    num_envs_per_run = 16  
    
    max_training_steps = 1_000_000 # Clean environment typically runs longer if needed
    target_score = -85.0  # Acrobot specific target
    
    convergence_times = []
    convergence_steps = []

    print("=== Starting E2E PPO (3M Params) Tests on ACROBOT IN CLEAN ENVIRONMENT ===")
    print(f"Target Score: {target_score} | Max Steps Allowed: {max_training_steps}\n")
    
    for idx, run_seed in enumerate(test_seeds):
        print(f"\n--- Running CLEAN Acrobot Seed {run_seed} ({idx + 1}/{len(test_seeds)}) with {num_envs_per_run} workers ---")
        
        # Dedicated directory for 3M param Acrobot clean models
        seed_dir = f"ppo_3m_clean_acrobot_results_seed_{run_seed}"
        os.makedirs(seed_dir, exist_ok=True)
        csv_save_path = os.path.join(seed_dir, "ppo_learning_curve.csv")
        model_save_path = os.path.join(seed_dir, "final_ppo_model.zip")
        
        vec_env = make_vec_env(
            env_id="Acrobot-v1", 
            n_envs=num_envs_per_run, 
            seed=run_seed,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=PixelAcrobot,
            env_kwargs={"render_mode": "rgb_array"}
        )
        
        vec_env = VecFrameStack(vec_env, n_stack=4)
        
        # ==========================================
        # 🌟 The 3M Parameter "Golden Ratio" Architecture 🌟
        # ==========================================
        custom_policy_kwargs = dict(
            features_extractor_kwargs=dict(features_dim=768),
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
        
        model = PPO("CnnPolicy", vec_env, policy_kwargs=custom_policy_kwargs, verbose=0, device="cuda", seed=run_seed,
                    n_steps=512, batch_size=256, learning_rate=3e-4)
        
        # Calculate and print precise parameter count
        total_params = sum(p.numel() for p in model.policy.parameters())
        trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        print(f"[*] PPO 3M Model Instantiated for CLEAN Acrobot!")
        print(f"[*] Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")
        # ==========================================

        # Attach the strict 10k interval CSV logger
        robust_cb = Strict10kLoggerCallback(save_path=csv_save_path, target_reward=target_score, check_freq=10000)
        start_time = time.time()
        
        model.learn(total_timesteps=max_training_steps, callback=robust_cb, progress_bar=False)
        
        train_duration = time.time() - start_time
        final_steps = model.num_timesteps
        
        # Force a final save at the very end just in case
        df_final = pd.DataFrame(robust_cb.history_data)
        df_final.to_csv(csv_save_path, index=False)
        
        model.save(model_save_path)
        
        print(f"[*] Final Result for Clean Seed {run_seed}: Stopped at {final_steps} steps in {train_duration:.2f} seconds")
        print(f"[*] Data strictly saved to: {csv_save_path}")
        
        convergence_times.append(train_duration)
        convergence_steps.append(final_steps)
        
        vec_env.close()

    print("\n" + "="*50)
    print("=== FINAL CONVERGENCE SUMMARY (ACROBOT 3M PARAMS, CLEAN) ===")
    print(f"Average Convergence Time: {np.mean(convergence_times):.2f} sec")
    print(f"Average Convergence Steps: {np.mean(convergence_steps):.0f} steps")
    print("="*50 + "\n")