#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LunarLander Dual Analysis: Clean Workspace & Single Trajectory Actions
# IEEE CEC Edition

import os
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
MODEL_PATH = "./best_model_101.npz"  
SINGLE_EPISODE_SEED = 42     
MAX_STEPS = 1000

# Boundary for the Grid
BOUNDS = {
    "X": (-1.0, 1.0),
    "Y": (0.0, 1.5)
}
GRID_RESOLUTION = 100

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
def generate_clean_workspace_grid(model):
    print("Generating uniform grid states for clean global workspace...")
    data_log = []
    
    xs = np.linspace(BOUNDS["X"][0], BOUNDS["X"][1], GRID_RESOLUTION)
    ys = np.linspace(BOUNDS["Y"][0], BOUNDS["Y"][1], GRID_RESOLUTION)
    
    for x in xs:
        for y in ys:
            # Control variables: perfect upright falling posture
            # This isolates the spatial (X, Y) decision boundary
            theta = 0.0
            v_x = 0.0
            v_y = -0.5  # Typical descent speed
            v_theta = 0.0
            v_norm = 0.5
            
            state = np.array([x, y, theta, v_x, v_y, v_theta, v_norm], dtype=np.float32)
            s_tensor = torch.tensor(state)
            
            with torch.no_grad():
                act = int(torch.argmax(model(s_tensor)).item())
                
            data_log.append({"X": x, "Y": y, "Action": act})
            
    return pd.DataFrame(data_log)

def generate_single_trajectory(model):
    print(f"Generating single trajectory (Seed {SINGLE_EPISODE_SEED}) via YOLO RPC...")
    trajectory = []
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=SINGLE_EPISODE_SEED)
    last_valid_state = np.zeros(7, dtype=np.float32)
    
    valid_rpc_received = False
    
    try:
        rpc = RPCClient()
        with rpc:
            rpc.reset()
            for t in tqdm(range(MAX_STEPS), desc="Simulating Trajectory"):
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
                
                trajectory.append({"X": state[0], "Y": state[1], "Action": act, "Step": t})
                
                # Sanity check for RPC connection failure
                if t == 10 and not valid_rpc_received:
                    print("\n[WARNING] YOLO RPC Server did not return valid data in the first 10 steps!")
                    print("[WARNING] The state is stuck at [0,0]. Please ensure server (port 6001) is running.\n")
                
                obs, reward, done, truncated, info = env.step(act)
                if done or truncated: break
    except Exception as e:
        print(f"RPC Error during trajectory generation: {e}")
        
    env.close()
    return pd.DataFrame(trajectory)

# ================= Plotting =================
def create_standard_legend():
    legend_elements = []
    for act in range(4):
        legend_elements.append(
            mlines.Line2D([0], [0], marker='s', color='w', markerfacecolor=ACTION_COLORS[act], 
                          markersize=8, label=ACTION_LABELS[act], alpha=0.8)
        )
    return legend_elements

def plot_global_workspace(df_bg):
    print("\nGenerating Global Workspace Area Plot...")
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    
    # Use markers to fill the grid, acting like a contour plot
    for act in range(4):
        subset = df_bg[df_bg['Action'] == act]
        if len(subset) > 0:
            ax.scatter(
                subset['X'], subset['Y'], 
                c=ACTION_COLORS[act], 
                s=25, alpha=0.4, marker='s', edgecolors='none', rasterized=True
            )

    ax.set_title("Global Policy Workspace Mapping", fontweight='bold')
    ax.set_xlabel("Horizontal Position (X)")
    ax.set_ylabel("Vertical Position (Y)")
    ax.set_xlim(BOUNDS["X"])
    ax.set_ylim(BOUNDS["Y"])
    ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
    
    ax.legend(handles=create_standard_legend(), loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    
    pdf_path = os.path.join(OUTPUT_DIR, "fig_workspace_global.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path}")

def plot_single_trajectory(df_traj):
    if df_traj.empty: return
        
    print("Generating Colored Trajectory Plot...")
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    
    # Connect the dots with a thin gray line to show path progression
    ax.plot(df_traj['X'], df_traj['Y'], color='gray', linewidth=0.8, linestyle='-', zorder=1, alpha=0.5)
    
    # Scatter points colored by action
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

    ax.set_title("Single Descent Action Trajectory", fontweight='bold')
    ax.set_xlabel("Horizontal Position (X)")
    ax.set_ylabel("Vertical Position (Y)")
    
    # Provide a buffer around the actual trajectory bounds
    x_min, x_max = df_traj['X'].min(), df_traj['X'].max()
    y_min, y_max = df_traj['Y'].min(), df_traj['Y'].max()
    
    # Check if the coordinates are stuck at 0 (RPC failure)
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
    
    pdf_path = os.path.join(OUTPUT_DIR, "fig_trajectory_actions.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[Fatal] Model file not found: {MODEL_PATH}")
    else:
        model = load_policy(MODEL_PATH)
        
        # 1. Plot clean decision boundary
        df_background = generate_clean_workspace_grid(model)
        plot_global_workspace(df_background)
        
        # 2. Plot real trajectory (Requires YOLO Server)
        df_trajectory = generate_single_trajectory(model)
        plot_single_trajectory(df_trajectory)