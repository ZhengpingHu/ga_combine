#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CartPole Evaluation Client: Cross-Validation Script
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

MODEL_PATHS = {
    101: "./best_model_101.npz", 
    102: "./best_model_102.npz",
    103: "./best_model_103.npz",
    104: "./best_model_104.npz",
    105: "./best_model_105.npz"
}

TEST_BASE_SEED = 106
TEST_ENV_COUNT = 100
MAX_STEPS = 500

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ==========================================
# [新增] 环境 Wrapper (高保真注入噪声，纯 NumPy 版)
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
            
            # 使用 NumPy 的 kron 进行简单的像素放大 (如果需要适配不同分辨率的话)
            # 由于 CartPole 就是 600x400，这里可以直接覆盖
            frame[bg_mask] = self.clutter_texture[bg_mask].astype(np.float32)

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
    # [修改] 接收 is_noisy 参数
    agent_id, weights, env_seed, rpc_host, rpc_port, authkey, is_noisy = args
    
    model = FixedNNPolicy()
    set_weights_vector(model, weights)
    
    total_reward = 0.0
    try:

        env = gym.make("CartPole-v1", render_mode="rgb_array")
        
        # [修改] 如果启用了 noisy，则挂载 Wrapper
        if is_noisy:
            env = VisuallyClutteredWrapper(env, gaussian_std=25.0, add_clutter=True)
            
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
    parser.add_argument("--outdir", default="evaluation_results")
    # [新增] 命令行参数：是否开启噪声
    parser.add_argument("--noisy", action="store_true", help="启用该参数以在噪声环境中测试")
    args = parser.parse_args()

    # 根据参数调整输出目录名称
    outdir = args.outdir + "_noisy" if args.noisy else args.outdir
    os.makedirs(outdir, exist_ok=True)

    test_seeds = generate_test_seeds(TEST_BASE_SEED, TEST_ENV_COUNT)

    tasks = []
    env_status = "NOISY" if args.noisy else "CLEAN"
    print(f"\nLoading models and preparing tasks... [Target Environment: {env_status}]")
    
    for agent_seed, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"[Warning] Model file not found: {model_path}. Skipping.")
            continue
            
        try:
            data = np.load(model_path)
            weights = data['weights']
            print(f" - Loaded Agent {agent_seed} from {model_path}")
            
            for env_seed in test_seeds:
                # [修改] 传递 args.noisy 到工作进程
                tasks.append((agent_seed, weights, env_seed, args.rpc_host, args.rpc_port, args.authkey, args.noisy))
                
        except Exception as e:
            print(f"[Error] Failed to load {model_path}: {e}")

    total_tasks = len(tasks)
    print(f"\nStarting Evaluation: {total_tasks} total episodes (5 Agents * 100 Envs)")
    print(f"Using {args.processes} processes.")


    results = []
    with mp.Pool(processes=args.processes) as pool:
        for res in tqdm(pool.imap_unordered(run_single_episode, tasks), total=total_tasks, desc="Evaluating"):
            results.append(res)


    df = pd.DataFrame(results)
    
    # 调整保存文件名
    csv_filename = "evaluation_data_106_noisy.csv" if args.noisy else "evaluation_data_106.csv"
    csv_path = os.path.join(outdir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"\nRaw data saved to: {csv_path}")

    summary = df.groupby("Agent_Source_Seed")["Reward"].agg(['mean', 'std', 'min', 'max'])
    print("\n=== Summary Statistics ===")
    print(summary)
    
    print(f"\nGenerating Violin Plot...")
    plt.figure(figsize=(12, 8))
    
    sns.violinplot(
        data=df, 
        x="Agent_Source_Seed", 
        y="Reward", 
        hue="Agent_Source_Seed",
        # 根据是否是 noisy 更改一下配色以作区分
        palette="magma" if args.noisy else "viridis",
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

    title_env = "NOISY" if args.noisy else "CLEAN"
    plt.title(f"Agent Robustness Evaluation in {title_env} Environment\n(Tested on 100 Unseen Environments generated from Seed {TEST_BASE_SEED})")
    plt.xlabel("Agent ID (Source Training Seed)")
    plt.ylabel("Total Reward")
    plt.grid(True, axis='y', alpha=0.3)
    
    # 调整图片保存名
    plot_filename = "agent_comparison_noisy_violin.png" if args.noisy else "agent_comparison_violin.png"
    plot_path = os.path.join(outdir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Violin plot saved to: {plot_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()