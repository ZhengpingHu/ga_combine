#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import multiprocessing as mp
from multiprocessing.connection import Client
from typing import Optional, Tuple
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))


class NNPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def get_weights_vector(m: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.data.flatten() for p in m.parameters()]).cpu().numpy()

def set_weights_vector(m: nn.Module, vec: np.ndarray):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

def uniform_crossover(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.rand(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(~mask, p1, p2)
    return c1, c2

def mutate(vec: np.ndarray, sigma: float) -> np.ndarray:
    return vec + np.random.randn(vec.size) * sigma

class RPCClient:
    def __init__(self, host: str, port: int, authkey: str = "yolo-rpc"):
        self.address = (host, port)
        self.authkey = authkey.encode("utf-8")
        self.conn: Optional[Client] = None

    def __enter__(self):
        try:
            self.conn = Client(self.address, authkey=self.authkey)
        except ConnectionRefusedError:
            print(f"[FATAL] Connection to RPC server {self.address} refused. Is the server running?")
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def reset(self):
        self.conn.send(("reset", None))
        _ = self.conn.recv()

    def update_full(self, frame_bgr: np.ndarray, steps: int = 0):
        self.conn.send(("update_full", (frame_bgr, steps)))
        ok, z = self.conn.recv()
        return z if ok else None

    def predict_only(self):
        self.conn.send(("predict_only", None))
        ok, z = self.conn.recv()
        return z if ok else None

def evaluate_individual(args):
    idx, weights, rpc_host, rpc_port, authkey = args
    model = NNPolicy()
    set_weights_vector(model, weights)

    total_reward = 0.0
    try:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        obs, _ = env.reset(seed=random.randint(0, 9999))
        
        with RPCClient(rpc_host, rpc_port, authkey) as rpc:
            rpc.reset()

            for t in range(1000):
                frame = env.render()
                if frame is None: break
                
                frame_bgr = frame[..., ::-1]

                state = rpc.update_full(frame_bgr, t)

                if state is None:
                    state = rpc.predict_only()
                if state is None:
                    state = np.zeros(7, dtype=np.float32)

                s = torch.tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    act = int(torch.argmax(model(s)).item())

                obs, reward, done, truncated, info = env.step(act)
                total_reward += reward

                if done or truncated:
                    break
        
        env.close()
        return idx, total_reward

    except Exception as e:
        return idx, -1000.0 


def save_best_model(run_dir: str, vec: np.ndarray):
    os.makedirs(run_dir, exist_ok=True)
    np.savez(os.path.join(run_dir, "best_model_vec.npz"), weights=vec)
    m = NNPolicy()
    set_weights_vector(m, vec)
    torch.save(m.state_dict(), os.path.join(run_dir, "best_model.pt"))

def save_metrics_csv(run_dir: str, gens, best_list, avg_list):
    path = os.path.join(run_dir, "metrics.csv")
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("generation,best,avg_top\n")
        for g, b, a in zip(gens, best_list, avg_list):
            f.write(f"{g},{b},{a}\n")

def plot_reward_curve(run_dir: str, all_gens, all_best, all_avg):
    if len(all_gens) == 0:
        return
    plt.figure(figsize=(8,5))
    plt.plot(all_gens, all_best, label="Best reward")
    plt.plot(all_gens, all_avg, label="AvgTop")
    plt.xlabel("Generation")
    plt.ylabel("Reward")
    plt.title("GA Reward Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out = os.path.join(run_dir, "reward_curve.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def run_ga(population_size, generations, elite_frac, sigma, processes,
           rpc_host, rpc_port, authkey,
           outdir: str, tag: Optional[str] = None):

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ga_{ts}" if not tag else f"ga_{ts}_{tag}"
    run_dir = os.path.join(outdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg = {
        "population": population_size,
        "generations": generations,
        "elite_frac": elite_frac,
        "init_sigma": sigma,
        "processes": processes,
        "rpc_host": rpc_host,
        "rpc_port": rpc_port,
        "authkey": authkey,
        "outdir": outdir,
        "run_dir": run_dir,
        "tag": tag or ""
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    model = NNPolicy()
    base_vec = get_weights_vector(model)
    pop = [mutate(base_vec, sigma=0.2) for _ in range(population_size)]

    best_global = -np.inf
    best_vec_global = None
    curve_gens, curve_best, curve_avg = [], [], []

    for gen in range(1, generations + 1):
        jobs = [(i, pop[i], rpc_host, rpc_port, authkey) for i in range(population_size)]
        
        scores = np.zeros(population_size)
        try:
            with mp.Pool(processes=processes) as pool:
                results = list(tqdm(pool.imap_unordered(evaluate_individual, jobs),
                                    total=len(jobs),
                                    desc=f"Generation {gen}/{generations}"))
            for idx, r in results:
                scores[idx] = r
        except Exception as e:
            print(f"\n[FATAL] Multiprocessing pool failed: {e}")
            print("Please ensure the RPC server is running and accessible.")
            plot_reward_curve(run_dir, curve_gens, curve_best, curve_avg)
            break

        elite_num = max(2, int(elite_frac * population_size))
        elite_idx = np.argsort(scores)[-elite_num:]
        elites = [pop[i] for i in elite_idx]

        best = float(scores[elite_idx[-1]])
        avg_top = float(scores[elite_idx].mean())
        print(f"[GEN {gen:03d}] Best={best:+.2f}  AvgTop{elite_num}={avg_top:+.2f}  Sigma={sigma:.3f}")

        curve_gens.append(gen)
        curve_best.append(best)
        curve_avg.append(avg_top)
        save_metrics_csv(run_dir, [gen], [best], [avg_top])
        plot_reward_curve(run_dir, curve_gens, curve_best, curve_avg)

        if best > best_global:
            best_global = best
            best_vec_global = pop[int(elite_idx[-1])]
            save_best_model(run_dir, best_vec_global)

        new_pop = elites.copy()
        while len(new_pop) < population_size:
            p1, p2 = random.sample(elites, 2)
            c1, c2 = uniform_crossover(p1, p2)
            new_pop.append(mutate(c1, sigma))
            if len(new_pop) < population_size:
                new_pop.append(mutate(c2, sigma))

        pop = new_pop
        sigma = max(0.01, sigma * 0.99)

    plot_reward_curve(run_dir, curve_gens, curve_best, curve_avg)
    if best_vec_global is not None:
        save_best_model(run_dir, best_vec_global)


def main():
    parser = argparse.ArgumentParser(description="GA Client for LunarLander with RPC state estimation")
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1", help="RPC server host")
    parser.add_argument("--rpc-port", type=int, default=6001, help="RPC server port")
    parser.add_argument("--authkey", type=str, default="yolo-rpc", help="RPC authentication key")
    parser.add_argument("--processes", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--population", type=int, default=120, help="Population size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--elite-frac", type=float, default=0.1, help="Fraction of elites to keep")
    parser.add_argument("--sigma", type=float, default=0.1, help="Initial mutation strength")

    parser.add_argument("--outdir", type=str, default="runs_ga", help="Directory to save logs/models")
    parser.add_argument("--tag", type=str, default="", help="Optional tag to append to run folder name")

    args = parser.parse_args()

    print(f"[INFO] Starting GA training with {args.processes} processes, "
          f"pop={args.population}, gen={args.generations}")
    print(f"📡 Connecting to RPC server at {args.rpc_host}:{args.rpc_port}")

    run_ga(args.population, args.generations, args.elite_frac, args.sigma,
           args.processes, args.rpc_host, args.rpc_port, args.authkey,
           outdir=args.outdir, tag=(args.tag or None))

if __name__ == "__main__":
    main()
