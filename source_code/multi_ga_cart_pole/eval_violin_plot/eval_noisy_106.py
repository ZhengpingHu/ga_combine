#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CartPole Evaluation Client: Cross-Validation Script IN NOISY ENVIRONMENT
# Task: Evaluate 5 Agents (Seeds 101-105) on 100 New Environments (Seed 106)

import os
import argparse
import random
import multiprocessing as mp
from multiprocessing.connection import Client
from typing import Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import cv2  # [NEW] Needed for visual noise injection

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 确保这里的路径指向你刚才加噪训练出来的那些 npz 文件！
MODEL_PATHS = {
    101: "./best_model_101.npz", # 请替换为实际路径
    102: "./best_model_102.npz", # 请替换为实际路径
    103: "./best_model_103.npz", # 请替换为实际路径
    104: "./best_model_104.npz", # 请替换为实际路径
    105: "./best_model_105.npz"  # 请替换为实际路径
}

TEST_BASE_SEED = 106
TEST_ENV_COUNT = 100
MAX_STEPS = 500

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ==========================================
# [NEW] Environment Wrapper for Visual Clutter
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        self.clutter_texture = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
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


class FixedNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def set_weights_vector(m: nn.Module, vec: np.ndarray):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

class RPCClient:
    def __init__(self, host, port, authkey="cartpole-rpc"):
        self.address = (host, port)
        self.authkey = authkey.encode("utf-8") if isinstance(authkey, str) else authkey
        self.conn: Optional[Client] = None
    def __enter__(self):
        try:
            self.conn = Client(self.address, authkey=self.authkey)
            return self
        except Exception as e:
            raise e
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()
    def reset(self):
        self.conn.send(("reset", None)); self.conn.recv()
    def infer(self, frame_bgr):
        self.conn.send(("infer", frame_bgr)); ok, res = self.conn.recv(); return res if ok else None

def generate_test_seeds(base_seed: int, count: int) -> List[int]:
    print(f"Generating {count} test seeds from base seed {base_seed}...")
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(low=0, high=2**31-1, size=count, dtype=np.int32).tolist()
    return seeds

def run_single_episode(args):
    agent_id, weights, env_seed, rpc_host, rpc_port, authkey = args
    
    model = FixedNNPolicy()
    set_weights_vector(model, weights)
    
    total_reward = 0.0
    try:
        # [MODIFIED]: Wrap the environment with visual noise
        base_env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = VisuallyClutteredWrapper(base_env, gaussian_std=25.0, add_clutter=True)
        obs, _ = env.reset(seed=int(env_seed))
        
        last_valid_state = np.zeros(4, dtype=np.float32)
        
        with RPCClient(rpc_host, rpc_port, authkey) as rpc:
            rpc.reset()
            for _ in range(MAX_STEPS):
                frame = env.render()
                if frame is None: break
                
                frame_bgr = frame[..., ::-1].copy()
                state = rpc.infer(frame_bgr)
                
                if state is None: state = last_valid_state
                else: last_valid_state = state
                
                s = torch.tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    act = int(torch.argmax(model(s)).item())
                
                obs, reward, done, truncated, info = env.step(act)
                total_reward += reward
                
                if done or truncated: break
        
        env.close()
        return {
            "Agent_Source_Seed": str(agent_id), 
            "Test_Env_Seed": env_seed,
            "Reward": total_reward
        }
        
    except Exception as e:
        print(f"Error on Agent {agent_id} / Seed {env_seed}: {e}")
        return {
            "Agent_Source_Seed": str(agent_id),
            "Test_Env_Seed": env_seed,
            "Reward": 0.0
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=6000)
    parser.add_argument("--authkey", default="cartpole-rpc")
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--outdir", default="evaluation_noisy_results") # Changed output directory
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    test_seeds = generate_test_seeds(TEST_BASE_SEED, TEST_ENV_COUNT)
    
    tasks = []
    print(f"\nLoading models and preparing tasks...")
    
    for agent_seed, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"[Warning] Model file not found: {model_path}. Skipping.")
            continue
            
        try:
            data = np.load(model_path)
            weights = data['weights']
            print(f" - Loaded Agent {agent_seed} from {model_path}")
            
            for env_seed in test_seeds:
                tasks.append((agent_seed, weights, env_seed, args.rpc_host, args.rpc_port, args.authkey))
                
        except Exception as e:
            print(f"[Error] Failed to load {model_path}: {e}")

    total_tasks = len(tasks)
    print(f"\nStarting NOISY Evaluation: {total_tasks} total episodes (5 Agents * 100 Envs)")
    print(f"Using {args.processes} processes.")

    results = []
    with mp.Pool(processes=args.processes) as pool:
        for res in tqdm(pool.imap_unordered(run_single_episode, tasks), total=total_tasks, desc="Evaluating"):
            results.append(res)

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.outdir, "evaluation_data_noisy_106.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRaw data saved to: {csv_path}")

    summary = df.groupby("Agent_Source_Seed")["Reward"].agg(['mean', 'std', 'min', 'max'])
    print("\n=== Summary Statistics (NOISY ENVIRONMENT) ===")
    print(summary)
    
    print(f"\nGenerating Violin Plot...")
    plt.figure(figsize=(12, 8))
    
    sns.violinplot(
        data=df, 
        x="Agent_Source_Seed", 
        y="Reward", 
        hue="Agent_Source_Seed",
        palette="magma", # Changed color palette to distinguish from clean eval
        inner="quartile", 
        legend=False
    )
    
    sns.stripplot(
        data=df, 
        x="Agent_Source_Seed", 
        y="Reward", 
        color="black", 
        alpha=0.2, 
        size=3,
        jitter=True
    )

    plt.title(f"Agent Robustness Evaluation in NOISY Environment\n(Tested on 100 Unseen Environments generated from Seed {TEST_BASE_SEED})")
    plt.xlabel("Agent ID (Source Training Seed)")
    plt.ylabel("Total Reward")
    plt.grid(True, axis='y', alpha=0.3)
    
    plot_path = os.path.join(args.outdir, "agent_comparison_noisy_violin.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Violin plot saved to: {plot_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()