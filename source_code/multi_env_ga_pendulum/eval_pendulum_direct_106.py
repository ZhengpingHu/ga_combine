#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pendulum Direct Env Evaluation: Seed 106 Pool
# Task: Evaluate 5 Champions on 100 New Environments (Seed 106) using internal state

import os
import argparse
import random
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import pandas as pd

# === 配置区域 ===
# 请根据实际存放位置修改模型路径
# 确保这些文件是刚才那个"直接读取环境信息"的训练脚本生成的
MODEL_PATHS = {
    101: "./best_model_final_101.npz",
    102: "./best_model_final_102.npz",
    103: "./best_model_final_103.npz",
    104: "./best_model_final_104.npz",
    105: "./best_model_final_105.npz"
}

TEST_GLOBAL_SEED = 106
TEST_POOL_SIZE = 100
MAX_STEPS = 200 # 与训练保持一致

# 性能优化
os.environ.setdefault("OMP_NUM_THREADS", "1")

# --- 网络定义 (必须与训练代码一致) ---
class PendulumNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Direct Env Input: 3 dims (cos_theta, sin_theta, theta_dot)
        # Hidden Layer: 16 (如前一个脚本修正版所示)
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pendulum 动作空间是 [-2.0, 2.0]
        return self.net(x) * 2.0

def set_weights_vector(m, vec):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

# --- 评估逻辑 (纯环境交互) ---
def evaluate_task(args):
    agent_id, weights, env_seed, max_steps = args
    model = PendulumNNPolicy()
    set_weights_vector(model, weights)
    total_reward = 0.0
    
    try:
        # 不需要 render_mode="rgb_array"，直接跑数值模拟更快
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset(seed=int(env_seed))
        
        for _ in range(max_steps):
            s_tensor = torch.tensor(obs, dtype=torch.float32)
            
            with torch.no_grad(): 
                action = model(s_tensor).item()
            
            # 环境直接返回下一个状态
            obs, reward, done, truncated, _ = env.step([action])
            total_reward += reward
            
            if done or truncated: break
            
        env.close()
        return {"Agent_Source_Seed": str(agent_id), "Test_Env_Seed": env_seed, "Reward": total_reward}
        
    except Exception as e:
        print(f"Error evaluating Agent {agent_id} on Seed {env_seed}: {e}")
        return {"Agent_Source_Seed": str(agent_id), "Test_Env_Seed": env_seed, "Reward": -2000.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--outdir", default="evaluation_results_pendulum_direct")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)

    # 1. 生成 100 个测试种子 (Global Seed 106)
    rng = np.random.default_rng(TEST_GLOBAL_SEED)
    test_seeds = rng.integers(0, 2**31-1, size=TEST_POOL_SIZE).tolist()

    # 2. 收集任务
    jobs = []
    print("Loading models and preparing validation pool...")
    models_found = 0
    
    for agent_seed, path in MODEL_PATHS.items():
        # 简单检查路径是否是占位符
        if "..." in path:
            print(f"[Skip] Path for Seed {agent_seed} looks like a placeholder.")
            continue
            
        if not os.path.exists(path):
            print(f"[Warning] Model file not found for Seed {agent_seed}: {path}")
            continue
            
        try:
            weights = np.load(path)['weights']
            models_found += 1
            for s in test_seeds:
                # 注意：任务参数里去掉了 rpc_port
                jobs.append((agent_seed, weights, s, MAX_STEPS))
        except Exception as e:
            print(f"[Error] Failed to load model {agent_seed}: {e}")

    if models_found == 0:
        print("No models loaded. Please update MODEL_PATHS in the script.")
        return

    # 3. 并行评估
    print(f"Starting cross-evaluation (Total episodes: {len(jobs)})...")
    results = []
    
    # 使用 imap_unordered 并行处理
    with mp.Pool(args.processes) as pool:
        for res in tqdm(pool.imap_unordered(evaluate_task, jobs), total=len(jobs)):
            results.append(res)

    # 4. 保存数据
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.outdir, "pendulum_direct_eval_seed106.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nEvaluation Complete. Data saved to: {csv_path}")
    
    # 打印统计摘要
    summary = df.groupby("Agent_Source_Seed")["Reward"].agg(['mean', 'std', 'min', 'max'])
    print("\n=== Summary Statistics (N=100 per Agent) ===")
    print(summary)

if __name__ == "__main__":
    main()