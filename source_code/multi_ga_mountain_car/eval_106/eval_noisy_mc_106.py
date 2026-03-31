#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MountainCar Discrete YOLO Evaluation: Seed 106 Pool
# Task: Evaluate 5 Champions using YOLO Visual Input on 100 New Environments

import os
import argparse
import multiprocessing as mp
from multiprocessing.connection import Client
import numpy as np
import cv2  # [新增] 用于处理背景纹理和图像缩放
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import time

MODEL_PATHS = {
    101: "./best_model_full_eval_101.npz",
    102: "./best_model_full_eval_102.npz",
    103: "./best_model_full_eval_103.npz",
    104: "./best_model_full_eval_104.npz",
    105: "./best_model_full_eval_105.npz"
}

TEST_GLOBAL_SEED = 106
TEST_POOL_SIZE = 100
MAX_STEPS = 200

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# [新增] 环境 Wrapper (高保真注入视觉噪声)
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        # MountainCar 原生分辨率是 600(宽) x 400(高)
        self.clutter_texture = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)

    def render(self):
        frame = self.env.render()
        if frame is None: return None
        frame = frame.astype(np.float32)

        if self.add_clutter:
            # MountainCar 的天空背景通常是纯色的 (>240)
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

# --- Policies & RPC ---

class NNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8, bias=True),
            nn.Tanh(), 
            nn.Linear(8, 3, bias=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def set_weights_vector(m, vec):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

class RPCClient:
    def __init__(self, host, port, authkey="mc-rpc"):
        self.address = (host, port)
        self.authkey = authkey.encode("utf-8")
        self.conn = None
        
    def __enter__(self):
        try:
            self.conn = Client(self.address, authkey=self.authkey)
            return self
        except Exception as e:
            return None
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            try:
                self.conn.close()
            except:
                pass

    def reset(self):
        if self.conn:
            try:
                self.conn.send(("reset", None))
                self.conn.recv() # Wait for ack
            except:
                pass

    def infer(self, frame_bgr):
        if self.conn:
            try:
                self.conn.send(("infer", frame_bgr))
                ok, res = self.conn.recv()
                return res if ok else None
            except:
                return None
        return None

def evaluate_task(args):
    # [修改] 解析新增加的 is_noisy 参数
    agent_id, weights, env_seed, rpc_host, rpc_port, authkey, is_noisy = args
    
    model = NNPolicy()
    set_weights_vector(model, weights)
    total_reward = 0.0
    
    try:
        env = gym.make("MountainCar-v0", render_mode="rgb_array")
        
        # [修改] 挂载噪声 Wrapper
        if is_noisy:
            env = VisuallyClutteredWrapper(env, gaussian_std=25.0, add_clutter=True)
            
        obs, _ = env.reset(seed=int(env_seed))
        
        last_valid_state = np.zeros(2, dtype=np.float32) 
        last_valid_state[:] = obs 

        try:
            rpc = RPCClient(rpc_host, rpc_port, authkey)
            with rpc:
                rpc.reset()
                
                for _ in range(MAX_STEPS):
                    frame = env.render()
                    if frame is None: break

                    frame_bgr = frame[..., ::-1].copy()
                    state_yolo = None
                    
                    if rpc.conn:
                        state_yolo = rpc.infer(frame_bgr)

                    if state_yolo is None:
                        state = last_valid_state
                    else:
                        state = np.array(state_yolo, dtype=np.float32)
                        last_valid_state = state

                    s_tensor = torch.tensor(state, dtype=torch.float32)
                    with torch.no_grad():
                        logits = model(s_tensor)
                        action = int(torch.argmax(logits).item())

                    obs, reward, done, truncated, _ = env.step(action)
                    total_reward += reward
                    
                    if done or truncated: break
                    
        except Exception as e:
            pass
            
        env.close()
        return {"Agent_Source_Seed": str(agent_id), "Test_Env_Seed": env_seed, "Reward": total_reward}
        
    except Exception as e:
        return {"Agent_Source_Seed": str(agent_id), "Test_Env_Seed": env_seed, "Reward": -200.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=6001, help="Port of the YOLO Server")
    parser.add_argument("--authkey", default="mc-rpc")
    parser.add_argument("--processes", type=int, default=16) 
    parser.add_argument("--outdir", default="evaluation_results_mc_yolo")
    # [新增] 噪声开关
    parser.add_argument("--noisy", action="store_true", help="Enable to evaluate in visually cluttered/noisy environment")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(TEST_GLOBAL_SEED)
    test_seeds = rng.integers(0, 2**31-1, size=TEST_POOL_SIZE).tolist()

    jobs = []
    env_status = "NOISY" if args.noisy else "CLEAN"
    print(f"Loading models and preparing YOLO evaluation tasks... [Env: {env_status}]")
    
    models_found = 0
    for agent_seed, path in MODEL_PATHS.items():
        if "..." in path: 
            print(f"[Skip] Path placeholder detected for {agent_seed}")
            continue
        if not os.path.exists(path):
            print(f"[Warning] Model not found: {path}")
            continue
            
        weights = np.load(path)['weights']
        models_found += 1
        for s in test_seeds:
            # [修改] 传递 args.noisy 到子进程
            jobs.append((agent_seed, weights, s, args.rpc_host, args.rpc_port, args.authkey, args.noisy))

    if models_found == 0:
        print("No models loaded. Check MODEL_PATHS.")
        return

    print(f"Starting visual evaluation on {len(jobs)} episodes.")
    print(f"Connecting to YOLO Server at {args.rpc_host}:{args.rpc_port}...")

    results = []
    with mp.Pool(args.processes) as pool:
        for res in tqdm(pool.imap_unordered(evaluate_task, jobs), total=len(jobs)):
            results.append(res)

    df = pd.DataFrame(results)
    
    # [修改] 动态生成文件名
    csv_filename = "mc_yolo_eval_seed106_noisy.csv" if args.noisy else "mc_yolo_eval_seed106_clean.csv"
    csv_path = os.path.join(args.outdir, csv_filename)
    df.to_csv(csv_path, index=False)
    
    print(f"\nEvaluation Complete. Data saved to: {csv_path}")

    summary = df.groupby("Agent_Source_Seed")["Reward"].agg(['mean', 'std', 'min', 'max'])
    print("\n=== Summary Statistics (Visual Input) ===")
    print(summary)

    success = df.groupby("Agent_Source_Seed").apply(lambda x: (x["Reward"] > -200).mean() * 100)
    print("\n=== Success Rate (%) ===")
    print(success)

if __name__ == "__main__":
    main()