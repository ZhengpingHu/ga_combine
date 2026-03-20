import os
import time
import gymnasium as gym
import numpy as np
import cv2
from collections import defaultdict
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium import spaces

# ==========================================
# 1. Global Profiler
# ==========================================
class Profiler:
    times = defaultdict(float)
    counts = defaultdict(int)

    @classmethod
    def start(cls):
        return time.perf_counter()

    @classmethod
    def stop(cls, name, start_time):
        cls.times[name] += (time.perf_counter() - start_time)
        cls.counts[name] += 1

    @classmethod
    def report(cls, total_train_time):
        print("\n" + "="*50)
        print("📊 TRUE PPO TIME BREAKDOWN (SINGLE PROCESS) 📊")
        print("="*50)
        
        env_step_time = cls.times.get("Env_Physics_Step", 0.0)
        env_render_time = cls.times.get("Env_Render_Resize", 0.0)
        
        total_env_time = env_step_time + env_render_time
        gpu_nn_time = total_train_time - total_env_time
        total_time_calc = total_env_time + gpu_nn_time

        print(f"Total Wall-Clock Time: {total_train_time:.2f} seconds\n")
        
        print(f"1. [CPU] Env Physics Step : {env_step_time:.2f}s ({env_step_time/total_time_calc*100:.1f}%)")
        print(f"2. [CPU] Env Render (Img) : {env_render_time:.2f}s ({env_render_time/total_time_calc*100:.1f}%)")
        print(f"3. [GPU] PPO NN (Fwd/Bwd) : {gpu_nn_time:.2f}s ({gpu_nn_time/total_time_calc*100:.1f}%)")
        print("-" * 50)
        print(f"Subtotal [CPU Simulation] : {total_env_time:.2f}s ({total_env_time/total_time_calc*100:.1f}%)")
        print(f"Subtotal [GPU Learning]   : {gpu_nn_time:.2f}s ({gpu_nn_time/total_time_calc*100:.1f}%)")
        print("="*50 + "\n")

# ==========================================
# 2. Profiled Pixel Wrapper
# ==========================================
class ProfiledPixelCartPole(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.img_size = (84, 84)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.uint8
        )

    def step(self, action):
        t0 = Profiler.start()
        obs, reward, terminated, truncated, info = self.env.step(action)
        Profiler.stop("Env_Physics_Step", t0)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        t0 = Profiler.start()
        img = self.env.render()
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        Profiler.stop("Env_Render_Resize", t0)
        return img

def make_profiled_env(env_id, seed, rank):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ProfiledPixelCartPole(env)
        env.action_space.seed(seed + rank)
        return env
    return _init

# ==========================================
# 3. Profiling Execution
# ==========================================
if __name__ == "__main__":
    test_seed = 101
    num_envs = 1  # 强制单线程运行以确保探针不丢失
    profiling_steps = 10_000 

    print(f"Starting TRUE Profiling Run ({profiling_steps} steps in DummyVecEnv)...")
    
    # CRITICAL CHANGE: 使用 DummyVecEnv 而不是 SubprocVecEnv
    vec_env = make_vec_env(
        env_id="CartPole-v1", 
        n_envs=num_envs, 
        seed=test_seed,
        vec_env_cls=DummyVecEnv, 
        wrapper_class=ProfiledPixelCartPole,
        env_kwargs={"render_mode": "rgb_array"}
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    model = PPO("CnnPolicy", vec_env, verbose=0, device="cuda", seed=test_seed,
                n_steps=512, batch_size=256)
    
    global_start = time.time()
    
    model.learn(total_timesteps=profiling_steps, progress_bar=True)
    
    total_train_duration = time.time() - global_start
    Profiler.report(total_train_duration)
    
    vec_env.close()