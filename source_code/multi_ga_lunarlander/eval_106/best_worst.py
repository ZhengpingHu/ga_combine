#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LunarLander Single Trajectory Extractor (IEEE CEC Edition)
# Runs 1000 episodes and extracts the absolute Best (> 200) and Worst (< 0) trajectories.

import os
# ================= CRITICAL FIX =================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# ================================================

import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
from multiprocessing.connection import Client
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm

# ================= Configuration =================
MODEL_PATH = "./best_model_104.npz"  
NUM_RUNS = 1000              
MAX_STEPS = 1000

# ================= IEEE Styling =================
OUTPUT_DIR = "analysis_plots_lunarlander"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try: plt.style.use('seaborn-v0_8-paper')
except: plt.style.use('seaborn-paper')

plt.rcParams.update({
    "font.family": "serif", 
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 10, 
    "font.size": 10, 
    "legend.fontsize": 8,
    "xtick.labelsize": 8, 
    "ytick.labelsize": 8,
    "axes.grid": True, 
    "grid.alpha": 0.3, 
    "grid.linestyle": "--",
    "axes.spines.top": False, 
    "axes.spines.right": False,
    "figure.constrained_layout.use": True, 
    "axes.unicode_minus": False
})

ACTION_COLORS = {
    0: '#B0BEC5',  # Gray  -> Do Nothing
    1: '#1E88E5',  # Blue  -> Left Engine
    2: '#E53935',  # Red   -> Main Engine
    3: '#43A047'   # Green -> Right Engine
}
ACTION_LABELS = {
    0: 'Do Nothing',
    1: 'Left Engine',
    2: 'Main Engine',
    3: 'Right Engine'
}

# ================= Policy Network & RPC =================
class NNPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, 4))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_policy(model_path):
    model = NNPolicy()
    data = np.load(model_path)
    vec = data['weights']
    
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n
    return model

class RPCClient:
    def __init__(self, host="127.0.0.1", port=6001, authkey="yolo-rpc"):
        self.address = (host, port)
        self.authkey = authkey.encode("utf-8")
        self.conn = None
    def __enter__(self):
        try:
            self.conn = Client(self.address, authkey=self.authkey)
            return self
        except Exception: return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()
    def reset(self):
        if self.conn:
            try: self.conn.send(("reset", None)); _ = self.conn.recv()
            except: pass
    def update_full(self, frame_bgr, steps=0):
        if self.conn:
            try:
                self.conn.send(("update_full", (frame_bgr, steps)))
                ok, z = self.conn.recv(); return z if ok else None
            except: return None
        return None

# ================= Data Generation =================
def extract_extreme_trajectories(model):
    print(f"Running {NUM_RUNS} episodes to find Best (> 200) and Worst (< 0) trajectories...")
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    best_reward = -np.inf
    worst_reward = np.inf
    best_trajectory = []
    worst_trajectory = []
    
    try:
        rpc = RPCClient()
        with rpc:
            if rpc.conn is None:
                print("[Fatal Error] Could not connect to YOLO RPC Server.")
                return pd.DataFrame(), pd.DataFrame()

            for seed in tqdm(range(NUM_RUNS), desc="Simulating 1000 Runs"):
                obs, _ = env.reset(seed=seed)
                rpc.reset()
                
                current_trajectory = []
                total_reward = 0.0
                last_valid_state = np.zeros(7, dtype=np.float32)
                valid_rpc_received = False
                
                for t in range(MAX_STEPS):
                    frame = env.render()
                    if frame is None: break
                    
                    frame_bgr = frame[..., ::-1]
                    state = None
                    
                    if rpc.conn:
                        state = rpc.update_full(frame_bgr, t)
                    
                    if state is None: 
                        state = last_valid_state
                    else: 
                        last_valid_state = state
                        valid_rpc_received = True
                    
                    s_tensor = torch.tensor(state, dtype=torch.float32)
                    with torch.no_grad(): 
                        act = int(torch.argmax(model(s_tensor)).item())
                    
                    current_trajectory.append({"X": state[0], "Y": state[1], "Action": act, "Step": t})
                    
                    if t == 10 and not valid_rpc_received:
                        print("\n[WARNING] YOLO RPC Server did not return valid data. Check connection.")
                        break
                    
                    obs, reward, done, truncated, info = env.step(act)
                    total_reward += reward
                    
                    if done or truncated: break
                
                # Update Best and Worst trackers
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_trajectory = current_trajectory
                
                if total_reward < worst_reward:
                    worst_reward = total_reward
                    worst_trajectory = current_trajectory

    except Exception as e:
        print(f"Error during trajectory extraction: {e}")
        
    env.close()
    
    print(f"\n--- Extraction Complete ---")
    print(f"Absolute Best Reward:  {best_reward:.2f}")
    print(f"Absolute Worst Reward: {worst_reward:.2f}")
    
    # Validation against user constraints
    if best_reward <= 200:
        print("[Notice] Did not find any run with reward > 200.")
    if worst_reward >= 0:
        print("[Notice] Did not find any run with reward < 0.")
        
    df_best = pd.DataFrame(best_trajectory)
    df_worst = pd.DataFrame(worst_trajectory)
    
    return df_best, best_reward, df_worst, worst_reward

# ================= Plotting =================
def create_standard_legend():
    legend_elements = []
    for act in range(4):
        legend_elements.append(
            mlines.Line2D([0], [0], marker='s', color='w', markerfacecolor=ACTION_COLORS[act], 
                          markersize=8, label=ACTION_LABELS[act], alpha=0.8)
        )
    return legend_elements

def plot_single_trajectory(df_traj, reward, is_best=True):
    if df_traj.empty: return
        
    tag = "Best" if is_best else "Worst"
    print(f"Generating Colored Trajectory Plot for {tag} Run (Reward: {reward:.2f})...")
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    
    # Continuous gray path line
    ax.plot(df_traj['X'], df_traj['Y'], color='gray', linewidth=0.8, linestyle='-', zorder=1, alpha=0.5)
    
    # Scatter colored by actions
    for act in range(4):
        subset = df_traj[df_traj['Action'] == act]
        if len(subset) > 0:
            ax.scatter(
                subset['X'], subset['Y'], 
                c=ACTION_COLORS[act], 
                s=20, zorder=2, edgecolors='none'
            )
            
    # Highlight Start and End points
    ax.scatter(df_traj['X'].iloc[0], df_traj['Y'].iloc[0], color='black', marker='s', s=40, zorder=3, label='Start')
    ax.scatter(df_traj['X'].iloc[-1], df_traj['Y'].iloc[-1], color='black', marker='X', s=50, zorder=3, label='End')

    ax.set_title(f"{tag} Descent Action Trajectory (Reward: {reward:.1f})", fontweight='bold')
    ax.set_xlabel("Horizontal Position (X)")
    ax.set_ylabel("Vertical Position (Y)")
    
    x_min, x_max = df_traj['X'].min(), df_traj['X'].max()
    y_min, y_max = df_traj['Y'].min(), df_traj['Y'].max()
    
    if x_min == 0 and x_max == 0 and y_min == 0 and y_max == 0:
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.text(0, 0.1, "RPC DATA MISSING", color='red', ha='center', fontweight='bold')
    else:
        ax.set_xlim(x_min - 0.2, x_max + 0.2)
        ax.set_ylim(y_min - 0.1, y_max + 0.2)
    
    ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
    
    legend_elements = create_standard_legend()
    legend_elements.extend([
        mlines.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=6, label='Start'),
        mlines.Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=6, label='End')
    ])
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    filename = f"fig_trajectory_{tag.lower()}_run.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[Fatal] Model file not found: {MODEL_PATH}")
    else:
        model = load_policy(MODEL_PATH)
        
        df_best, best_val, df_worst, worst_val = extract_extreme_trajectories(model)
        
        plot_single_trajectory(df_best, best_val, is_best=True)
        plot_single_trajectory(df_worst, worst_val, is_best=False)