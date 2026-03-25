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
# 1. Combined Wrapper: Visual Clutter + Pixel Resizing
# ==========================================
class NoisyPixelCartPole(gym.ObservationWrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        # Standard input size for Nature CNN (84x84)
        self.img_size = (84, 84)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.uint8
        )
        
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        # Pre-generate a static random noise texture for the background
        self.clutter_texture = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)

    def observation(self, obs):
        # 1. Render the raw RGB frame
        frame = self.env.render()
        if frame is None:
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            
        frame = frame.astype(np.float32)

        # 2. Inject Dynamic Background Clutter
        if self.add_clutter:
            # CartPole's background is nearly pure white. Mask pixels > 240.
            bg_mask = np.all(frame > 240, axis=-1)
            
            # Roll texture to create a dynamic/moving background effect
            self.clutter_texture = np.roll(self.clutter_texture, shift=3, axis=0)
            self.clutter_texture = np.roll(self.clutter_texture, shift=7, axis=1)
            
            h, w = frame.shape[:2]
            texture_resized = cv2.resize(self.clutter_texture, (w, h))
            
            # Replace pure white background with clutter
            frame[bg_mask] = texture_resized[bg_mask].astype(np.float32)

        # 3. Inject Gaussian Sensor Noise
        if self.gaussian_std > 0:
            noise = np.random.normal(0, self.gaussian_std, frame.shape)
            frame += noise

        # Clip back to valid RGB range [0, 255]
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # 4. Resize to match CNN input requirements
        img = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
        return img

# ==========================================
# 2. Advanced Callback: Early Stop, CSV Saving & Real-time Logging
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
            # Calculate rolling average of recent 20 episodes
            recent_avg = np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) > 0 else 0.0
            elapsed_time = time.time() - self.start_time
            print(f"    [Progress] Steps: {self.num_timesteps} | Elapsed: {elapsed_time:.1f}s | Recent 20-Ep Avg Reward: {recent_avg:.1f}")
            
            # Save learning curve dynamically
            df = pd.DataFrame({
                'Timestep': self.timesteps,
                'Reward': self.episode_rewards
            })
            df.to_csv(self.save_path, index=False)
            
            self.last_print_step = self.num_timesteps
            
            # Trigger Early Stopping ONLY if we have at least 20 episodes
            if len(self.episode_rewards) >= 20 and recent_avg >= self.target_reward:
                print(f"\n    [SUCCESS] Target reward {self.target_reward} reached at step {self.num_timesteps}!")
                print(f"    [SUCCESS] Time taken to converge: {elapsed_time:.2f} seconds.")
                return False  # Stop the training loop
            
        return True

# ==========================================
# 3. Main Experiment Execution
# ==========================================
if __name__ == "__main__":
    # Fixed seeds for rigorous reproducibility
    test_seeds = [101, 102, 103, 104, 105]
    num_envs_per_run = 16  # Matches the 16 parallel processes of the GA evaluation
    
    # 4.5 Million steps ~ 2.5 Hours (Equivalent to YOLO Pretrain + GA Evolution time)
    max_training_steps = 1_000_000 
    target_score = 495.0 
    
    convergence_times = []
    convergence_steps = []

    print("=== Starting E2E PPO Vision Baseline (Strict Time-Equivalent) Tests IN NOISY ENVIRONMENT ===")
    print(f"Target Score: {target_score} | Max Steps Allowed: {max_training_steps}\n")
    
    for idx, run_seed in enumerate(test_seeds):
        print(f"\n--- Running NOISY Seed {run_seed} ({idx + 1}/{len(test_seeds)}) with {num_envs_per_run} parallel workers ---")
        
        # Create a dedicated directory for this seed (Modified name to separate from clean baseline)
        seed_dir = f"ppo_noisy_results_seed_{run_seed}"
        os.makedirs(seed_dir, exist_ok=True)
        csv_save_path = os.path.join(seed_dir, "ppo_learning_curve.csv")
        model_save_path = os.path.join(seed_dir, "final_ppo_model.zip")
        
        # Create vectorized environment using the newly combined NoisyPixelCartPole wrapper
        vec_env = make_vec_env(
            env_id="CartPole-v1", 
            n_envs=num_envs_per_run, 
            seed=run_seed,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=NoisyPixelCartPole,
            env_kwargs={"render_mode": "rgb_array"}
        )
        
        # CRITICAL FIX: Stack 4 consecutive frames so the CNN can infer velocity
        vec_env = VecFrameStack(vec_env, n_stack=4)
        
        # Initialize PPO with CNN policy and optimized hyperparams for 16 envs
        model = PPO("CnnPolicy", vec_env, verbose=0, device="cuda", seed=run_seed,
                    n_steps=512, batch_size=256, learning_rate=3e-4)
        
        # Attach our custom Robust Logger callback
        robust_cb = RobustLoggerCallback(save_path=csv_save_path, target_reward=target_score, check_freq=10000)
        start_time = time.time()
        
        # Execute training
        model.learn(total_timesteps=max_training_steps, callback=robust_cb, progress_bar=False)
        
        train_duration = time.time() - start_time
        final_steps = model.num_timesteps
        
        # Save final learning curve
        df_final = pd.DataFrame({'Timestep': robust_cb.timesteps, 'Reward': robust_cb.episode_rewards})
        df_final.to_csv(csv_save_path, index=False)
        
        # ==========================================
        # 🌟 SAVE THE TRAINED PPO MODEL 🌟
        # ==========================================
        model.save(model_save_path)
        
        print(f"[*] Final Result for Noisy Seed {run_seed}: Stopped at {final_steps} steps in {train_duration:.2f} seconds")
        print(f"[*] Data strictly saved to: {csv_save_path}")
        print(f"[*] Model successfully saved to: {model_save_path}")
        
        convergence_times.append(train_duration)
        convergence_steps.append(final_steps)
        
        # Close environments to free memory before starting the next seed
        vec_env.close()

    # Print ultimate summary
    print("\n" + "="*50)
    print("=== FINAL CONVERGENCE SUMMARY (NOISY ENVIRONMENT) ACROSS 5 SEEDS ===")
    print(f"Average Convergence Time: {np.mean(convergence_times):.2f} sec (±{np.std(convergence_times):.2f})")
    print(f"Average Convergence Steps: {np.mean(convergence_steps):.0f} steps (±{np.std(convergence_steps):.0f})")
    print("="*50 + "\n")