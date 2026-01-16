#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pendulum Cross-Evaluation: Seed 106 Pool
# Task: Evaluate 5 Champions on 100 New Environments (Seed 106)

import os
import argparse
import random
import multiprocessing as mp
from multiprocessing.connection import Client
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import pandas as pd

MODEL_PATHS = {
    101: "./best_model_101.npz",
    102: "./best_model_102.npz",
    103: "./best_model_103.npz",
    104: "./best_model_104.npz",
    105: "./best_model_105.npz"
}

TEST_GLOBAL_SEED = 106
TEST_POOL_SIZE = 100
MAX_STEPS = 200

os.environ.setdefault("OMP_NUM_THREADS", "1")

class PendulumNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * 2.0

def set_weights_vector(m, vec):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

class RPCClient:
    def __init__(self, port, authkey=b"pendulum-rpc"):
        self.conn = Client(('127.0.0.1', port), authkey=authkey)
    def reset(self): self.conn.send(("reset", None)); self.conn.recv()
    def infer(self, frame):
        self.conn.send(("infer", frame)); ok, res = self.conn.recv()
        return res if ok else None
    def close(self): self.conn.close()

def evaluate_task(args):
    agent_id, weights, env_seed, rpc_port, max_steps = args
    model = PendulumNNPolicy()
    set_weights_vector(model, weights)
    total_reward = 0.0
    
    try:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        obs, _ = env.reset(seed=int(env_seed))
        last_state = np.zeros(4, dtype=np.float32)
        
        rpc = RPCClient(rpc_port)
        rpc.reset()
        
        for _ in range(max_steps):
            frame = env.render()
            state = rpc.infer(frame[..., ::-1].copy())
            if state is None: state = last_state
            else: last_state = state
            
            s_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad(): 
                action = model(s_tensor).item()
            
            _, reward, done, truncated, _ = env.step([action])
            total_reward += reward
            if done or truncated: break
            
        rpc.close()
        env.close()
        return {"Agent_Source_Seed": agent_id, "Env_Seed": env_seed, "Reward": total_reward}
    except Exception:
        return {"Agent_Source_Seed": agent_id, "Env_Seed": env_seed, "Reward": -2000.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-port", type=int, default=6000)
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--outdir", default="evaluation_results_pendulum")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(TEST_GLOBAL_SEED)
    test_seeds = rng.integers(0, 2**31-1, size=TEST_POOL_SIZE).tolist()

    jobs = []
    print("Loading models and preparing validation pool...")
    for agent_seed, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"[Warning] Path not found for Seed {agent_seed}: {path}")
            continue
        weights = np.load(path)['weights']
        for s in test_seeds:
            jobs.append((agent_seed, weights, s, args.rpc_port, MAX_STEPS))

    print(f"Starting cross-evaluation (Total episodes: {len(jobs)})...")
    results = []
    with mp.Pool(args.processes) as pool:
        for res in tqdm(pool.imap_unordered(evaluate_task, jobs), total=len(jobs)):
            results.append(res)

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.outdir, "pendulum_eval_seed106.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nEvaluation Complete. Data saved to: {csv_path}")
    summary = df.groupby("Agent_Source_Seed")["Reward"].agg(['mean', 'std', 'min', 'max'])
    print("\nSummary Table:")
    print(summary)

if __name__ == "__main__":
    main()