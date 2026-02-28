#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LunarLander Evaluation Client: Cross-Validation & IEEE Stripplot
# Task: Evaluate 5 Champions (101-105) on 100 New Environments (Seed 106)

import os
import argparse
import multiprocessing as mp
from multiprocessing.connection import Client
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- IEEE Style Matplotlib Config ----------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300
# --------------------------------------------------------------

# Matches the npz format from your training output
MODEL_PATHS = {
    101: "./best_model_101.npz", 
    102: "./best_model_102.npz",
    103: "./best_model_103.npz",
    104: "./best_model_104.npz",
    105: "./best_model_105.npz"
}

TEST_BASE_SEED = 106
TEST_ENV_COUNT = 100
MAX_STEPS = 1000

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- Classes and functions extracted directly from your training code ---
class NNPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, 4))
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
    def __init__(self, host, port, authkey="yolo-rpc"):
        self.address = (host, port); self.authkey = authkey.encode("utf-8"); self.conn: Optional[Client] = None
    def __enter__(self):
        try:
            self.conn = Client(self.address, authkey=self.authkey); return self
        except Exception:
            return None # Fail gracefully
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()
    def reset(self):
        if self.conn:
            try:
                self.conn.send(("reset", None)); _ = self.conn.recv()
            except: pass
    def update_full(self, frame_bgr, steps=0):
        if self.conn:
            try:
                self.conn.send(("update_full", (frame_bgr, steps))); ok, z = self.conn.recv(); return z if ok else None
            except: return None
        return None
    def predict_only(self):
        if self.conn:
            try:
                self.conn.send(("predict_only", None)); ok, z = self.conn.recv(); return z if ok else None
            except: return None
        return None
# ---------------------------------------------------------

def generate_test_seeds(base_seed: int, count: int) -> List[int]:
    print(f"Generating {count} test seeds from base seed {base_seed}...")
    rng = np.random.default_rng(base_seed)
    return rng.integers(low=0, high=2**31-1, size=count, dtype=np.int32).tolist()

def run_single_episode(args):
    agent_id, weights, env_seed, rpc_host, rpc_port, authkey = args
    
    model = NNPolicy()
    set_weights_vector(model, weights)
    
    total_reward = 0.0
    try:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        obs, _ = env.reset(seed=int(env_seed))
        last_valid_state = np.zeros(7, dtype=np.float32)
        
        # 100% replication of the training interaction logic
        try:
            rpc = RPCClient(rpc_host, rpc_port, authkey)
            with rpc:
                rpc.reset()
                for t in range(MAX_STEPS):
                    frame = env.render()
                    if frame is None: break
                    
                    frame_bgr = frame[..., ::-1] # Convert RGB to BGR for OpenCV/YOLO
                    
                    state = None
                    if rpc.conn:
                        state = rpc.update_full(frame_bgr, t)
                        if state is None: state = rpc.predict_only()
                    
                    if state is None: 
                        state = last_valid_state # Fallback logic
                    else: 
                        last_valid_state = state
                    
                    s = torch.tensor(state, dtype=torch.float32)
                    with torch.no_grad(): act = int(torch.argmax(model(s)).item())
                    
                    obs, reward, done, truncated, info = env.step(act)
                    total_reward += reward
                    if done or truncated: break
        except Exception:
            pass # Continue even if RPC fails
            
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
            "Reward": -1000.0 # Replicate the error penalty from your code
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=6001) # Strictly match port 6001
    parser.add_argument("--authkey", default="yolo-rpc")      # Strictly match yolo-rpc
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--outdir", default="evaluation_results")
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
            weights = data['weights'] # Strictly read matching np.savez(..., weights=...)
            print(f" - Loaded Agent {agent_seed} from {model_path}")
            
            for env_seed in test_seeds:
                tasks.append((agent_seed, weights, env_seed, args.rpc_host, args.rpc_port, args.authkey))
                
        except Exception as e:
            print(f"[Error] Failed to load {model_path}: {e}")

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No tasks to run. Check if .npz files exist.")
        return

    print(f"\nStarting Evaluation: {total_tasks} total episodes (Agents * {TEST_ENV_COUNT} Envs)")
    print(f"Using {args.processes} processes.")

    results = []
    with mp.Pool(processes=args.processes) as pool:
        for res in tqdm(pool.imap_unordered(run_single_episode, tasks), total=total_tasks, desc="Evaluating"):
            results.append(res)

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.outdir, "evaluation_data_106.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRaw data saved to: {csv_path}")

    summary = df.groupby("Agent_Source_Seed")["Reward"].agg(['mean', 'std', 'min', 'max'])
    print("\n=== Summary Statistics ===")
    print(summary)
    
    # ---------------- IEEE Style Dot Plot ----------------
    print(f"\nGenerating IEEE Style Dot Plot...")
    
    # IEEE single column standard width is approx 3.5 inches
    fig, ax = plt.subplots(figsize=(3.5, 2.6)) 
    
    # Keep the underlying violin plot contour, overlay scatter points
    sns.violinplot(
        data=df, 
        x="Agent_Source_Seed", 
        y="Reward", 
        color="lightgray",  # IEEE preferred grayscale background
        inner=None,         # Remove inner boxplot to make room for scatter
        linewidth=0.8,
        ax=ax
    )
    
    sns.stripplot(
        data=df, 
        x="Agent_Source_Seed", 
        y="Reward", 
        color="black",      
        alpha=0.5,          
        size=3,             
        jitter=True,        
        ax=ax,
        zorder=5            # Ensure scatter points are above the violin plot
    )

    ax.set_title(f"Agent Robustness on {TEST_ENV_COUNT} Unseen Environments")
    ax.set_xlabel("Agent ID (Training Seed)")
    ax.set_ylabel("Total Reward")
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    plot_path = os.path.join(args.outdir, "agent_reward_distribution_ieee.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PDF plot saved to: {plot_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()