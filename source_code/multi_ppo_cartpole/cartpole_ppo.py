import os
import gymnasium as gym
import numpy as np
import cv2
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import spaces

# ==========================================
# 1. Pixel Wrapper for End-to-End Vision
# ==========================================
class PixelCartPole(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Standard input size for Nature CNN
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

def make_env(env_id, seed, rank):
    def _init():
        # Must enable rgb_array for visual extraction
        env = gym.make(env_id, render_mode="rgb_array")
        env = Monitor(env) 
        env = PixelCartPole(env)
        # Offset the seed for each parallel worker
        env.action_space.seed(seed + rank)
        return env
    return _init

# ==========================================
# 2. Advanced Callback: Early Stop & Real-time CSV Saving
# ==========================================
class RobustLoggerCallback(BaseCallback):
    def __init__(self, save_path, target_reward=495.0, check_freq=10000, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.timesteps = []
        self.target_reward = target_reward
        self.check_freq = check_freq
        self.last_print_step = 0
        self.start_time = None
        self.save_path = save_path # Path to save intermediate CSV

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Extract reward info if an episode just finished
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.timesteps.append(self.num_timesteps)
        
        # Real-time progress display, CSV saving, and Early Stop check
        if self.num_timesteps - self.last_print_step >= self.check_freq:
            # 1. Calculate rolling average
            recent_avg = np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) > 0 else 0.0
            elapsed_time = time.time() - self.start_time
            print(f"    [Progress] Steps: {self.num_timesteps} | Elapsed: {elapsed_time:.1f}s | Recent 20-Ep Avg Reward: {recent_avg:.1f}")
            
            # 2. Intermediate Data Saving (Overwrite CSV safely)
            df = pd.DataFrame({
                'Timestep': self.timesteps,
                'Reward': self.episode_rewards
            })
            df.to_csv(self.save_path, index=False)
            
            self.last_print_step = self.num_timesteps
            
            # 3. Trigger Early Stopping if target is reached
            if recent_avg >= self.target_reward:
                print(f"\n    [SUCCESS] Target reward {self.target_reward} reached at step {self.num_timesteps}!")
                print(f"    [SUCCESS] Time taken to converge: {elapsed_time:.2f} seconds.")
                return False  # Returning False stops the training
            
        return True

# ==========================================
# 3. Main Experiment Execution
# ==========================================
if __name__ == "__main__":
    # Define exact seeds for rigorous reproducibility
    test_seeds = [101, 102, 103, 104, 105]
    num_envs_per_run = 16  # Use 16 concurrent CPU workers, same with YOLO frame
    
    # Set a massive max step count; Callback will halt it when solved
    max_training_steps = 2_000_000 
    target_score = 495.0 
    
    convergence_times = []
    convergence_steps = []

    print("=== Starting E2E PPO Vision Baseline (Time-to-Convergence) Tests ===")
    print(f"Target Score: {target_score} | Max Steps Allowed: {max_training_steps}\n")
    
    for idx, run_seed in enumerate(test_seeds):
        print(f"\n--- Running Seed {run_seed} ({idx + 1}/{len(test_seeds)}) with {num_envs_per_run} parallel workers ---")
        
        # Create a dedicated directory for this seed
        seed_dir = f"ppo_results_seed_{run_seed}"
        os.makedirs(seed_dir, exist_ok=True)
        csv_save_path = os.path.join(seed_dir, "ppo_learning_curve.csv")
        
        # True multiprocessing using SubprocVecEnv
        vec_env = make_vec_env(
            env_id="CartPole-v1", 
            n_envs=num_envs_per_run, 
            seed=run_seed,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=PixelCartPole,
            env_kwargs={"render_mode": "rgb_array"}
        )
        
        # Initialize PPO with CNN policy
        model = PPO("CnnPolicy", vec_env, verbose=0, device="cuda", seed=run_seed)
        
        # Log parameter counts only on the first run
        if idx == 0:
            total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
            print(f"[*] Architecture Parameter Analysis:")
            print(f"    - Total Trainable Parameters: {total_params:,}")
            print("-" * 50)

        # Attach our custom Robust Logger callback
        robust_cb = RobustLoggerCallback(save_path=csv_save_path, target_reward=target_score, check_freq=10000)
        start_time = time.time()
        
        # Execute training
        model.learn(total_timesteps=max_training_steps, callback=robust_cb, progress_bar=False)
        
        train_duration = time.time() - start_time
        final_steps = model.num_timesteps
        
        # Final explicit save to catch any remaining steps since last check
        df_final = pd.DataFrame({'Timestep': robust_cb.timesteps, 'Reward': robust_cb.episode_rewards})
        df_final.to_csv(csv_save_path, index=False)
        
        print(f"[*] Final Result for Seed {run_seed}: Stopped at {final_steps} steps in {train_duration:.2f} seconds")
        print(f"[*] Data strictly saved to: {csv_save_path}")
        
        convergence_times.append(train_duration)
        convergence_steps.append(final_steps)

    # Print ultimate summary
    print("\n" + "="*50)
    print("=== FINAL CONVERGENCE SUMMARY ACROSS 5 SEEDS ===")
    print(f"Average Convergence Time: {np.mean(convergence_times):.2f} sec (±{np.std(convergence_times):.2f})")
    print(f"Average Convergence Steps: {np.mean(convergence_steps):.0f} steps (±{np.std(convergence_steps):.0f})")
    print("="*50 + "\n")