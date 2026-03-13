#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LunarLander Vision System Analysis Tool (IEEE CEC Edition)
# Includes Auto-Calibration using pure Numpy (No sklearn required).

import os
# ================= CRITICAL FIX =================
# Must be set BEFORE importing numpy, torch, or gym
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# ================================================

import math
import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
from multiprocessing.connection import Client
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from tqdm import tqdm

# ================= Configuration =================
MODEL_PATH = "./best_model_101.npz"  
SAMPLE_COUNT = 2000          
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

COLOR_SCATTER = "#1f77b4"  
COLOR_IDEAL   = "black"

# ================= Policy & RPC Definitions =================
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

# ================= Data Sampling & Auto-Calibration =================
def generate_lunarlander_calibrated_data(model):
    print(f"Sampling {SAMPLE_COUNT} points from live environment using YOLO RPC...")
    raw_data = []
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    pbar = tqdm(total=SAMPLE_COUNT)
    collected = 0
    seed = 42
    
    try:
        rpc = RPCClient()
        with rpc:
            while collected < SAMPLE_COUNT:
                obs, _ = env.reset(seed=seed)
                seed += 1
                rpc.reset()
                last_valid_state = np.zeros(7, dtype=np.float32)
                
                for t in range(MAX_STEPS):
                    if collected >= SAMPLE_COUNT: break
                    
                    frame = env.render()
                    if frame is None: break
                    
                    frame_bgr = frame[..., ::-1]
                    pred_state = None
                    
                    if rpc.conn:
                        pred_state = rpc.update_full(frame_bgr, t)
                    
                    if pred_state is None: 
                        pred_state = last_valid_state
                    else: 
                        last_valid_state = pred_state
                        
                    true_x = obs[0]
                    true_y = obs[1]
                    true_theta = obs[4]
                    
                    raw_pred_x = pred_state[0]
                    raw_pred_y = pred_state[1]
                    raw_pred_theta = pred_state[2]
                    
                    raw_data.append({
                        "True_X": true_x, "True_Y": true_y, "True_Theta": true_theta,
                        "Raw_Pred_X": raw_pred_x, "Raw_Pred_Y": raw_pred_y, "Raw_Pred_Theta": raw_pred_theta
                    })
                    collected += 1
                    pbar.update(1)
                    
                    s_tensor = torch.tensor(pred_state, dtype=torch.float32)
                    with torch.no_grad(): 
                        act = int(torch.argmax(model(s_tensor)).item())
                    
                    obs, reward, done, truncated, info = env.step(act)
                    if done or truncated: break
                    
    except Exception as e:
        print(f"\n[Fatal Error] Data collection stopped: {e}")
        
    pbar.close()
    env.close()
    
    df = pd.DataFrame(raw_data)
    
    print("\nApplying Auto-Calibration (Numpy polyfit 1D)...")
    
    # Fit linear models: True = m * Raw_Pred + c
    coef_x = np.polyfit(df['Raw_Pred_X'], df['True_X'], 1)
    coef_y = np.polyfit(df['Raw_Pred_Y'], df['True_Y'], 1)
    coef_theta = np.polyfit(df['Raw_Pred_Theta'], df['True_Theta'], 1)
    
    df['Calib_Pred_X'] = df['Raw_Pred_X'] * coef_x[0] + coef_x[1]
    df['Calib_Pred_Y'] = df['Raw_Pred_Y'] * coef_y[0] + coef_y[1]
    df['Calib_Pred_Theta'] = df['Raw_Pred_Theta'] * coef_theta[0] + coef_theta[1]
    
    df['Err_X'] = df['Calib_Pred_X'] - df['True_X']
    df['Err_Y'] = df['Calib_Pred_Y'] - df['True_Y']
    
    # Angular error considering periodicity
    df['Err_Theta'] = np.arctan2(np.sin(df['Calib_Pred_Theta'] - df['True_Theta']), 
                                 np.cos(df['Calib_Pred_Theta'] - df['True_Theta']))
    
    df['Total_Pos_Err'] = np.sqrt(df['Err_X']**2 + df['Err_Y']**2)
    
    return df

# ================= Plotting =================
def plot_lunarlander_analysis(df):
    if len(df) == 0: 
        print("No data collected. Is the YOLO RPC server running?")
        return
        
    print("\nGenerating Calibrated Vision Analysis Plots...")

    # --- Plot A: Spatial Error Map ---
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    
    triang = tri.Triangulation(df["True_X"], df["True_Y"])
    cnt = ax.tricontourf(triang, df["Total_Pos_Err"], levels=20, cmap="Reds", vmin=0)
    
    cbar = fig.colorbar(cnt, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Calibrated Position Error', rotation=270, labelpad=15)
    
    ax.set_title("Calibrated Spatial Error Map", fontweight='bold')
    ax.set_xlabel("True Horizontal Position (X)")
    ax.set_ylabel("True Vertical Position (Y)")
    
    ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    
    spatial_path = os.path.join(OUTPUT_DIR, "fig_lunarlander_spatial_error.pdf")
    plt.savefig(spatial_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {spatial_path}")

    # --- Plot B: Linearity ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.0, 3.2), dpi=300)
    
    # 1. X Position
    ax1.scatter(df["True_X"], df["Calib_Pred_X"], s=2, c=COLOR_SCATTER, alpha=0.5, rasterized=True)
    ax1.plot([-2, 2], [-2, 2], c=COLOR_IDEAL, ls='--', lw=1)
    ax1.set_title("Horizontal (X) Linearity")
    ax1.set_xlabel("True X")
    ax1.set_ylabel("Calibrated Pred X")
    ax1.set_xlim(df["True_X"].min()-0.1, df["True_X"].max()+0.1)
    ax1.set_ylim(df["True_X"].min()-0.1, df["True_X"].max()+0.1)
    mae_x = abs(df['Err_X']).mean()
    ax1.text(0.05, 0.9, f"MAE: {mae_x:.3f}", transform=ax1.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 2. Y Position
    ax2.scatter(df["True_Y"], df["Calib_Pred_Y"], s=2, c=COLOR_SCATTER, alpha=0.5, rasterized=True)
    ax2.plot([-1, 2], [-1, 2], c=COLOR_IDEAL, ls='--', lw=1)
    ax2.set_title("Vertical (Y) Linearity")
    ax2.set_xlabel("True Y")
    ax2.set_ylabel("Calibrated Pred Y")
    ax2.set_xlim(df["True_Y"].min()-0.1, df["True_Y"].max()+0.1)
    ax2.set_ylim(df["True_Y"].min()-0.1, df["True_Y"].max()+0.1)
    mae_y = abs(df['Err_Y']).mean()
    ax2.text(0.05, 0.9, f"MAE: {mae_y:.3f}", transform=ax2.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 3. Theta (Angle)
    ax3.scatter(df["True_Theta"], df["Calib_Pred_Theta"], s=2, c=COLOR_SCATTER, alpha=0.5, rasterized=True)
    ax3.plot([-3, 3], [-3, 3], c=COLOR_IDEAL, ls='--', lw=1)
    ax3.set_title("Orientation (Theta) Linearity")
    ax3.set_xlabel("True Theta (rad)")
    ax3.set_ylabel("Calibrated Pred Theta")
    ax3.set_xlim(df["True_Theta"].min()-0.1, df["True_Theta"].max()+0.1)
    ax3.set_ylim(df["True_Theta"].min()-0.1, df["True_Theta"].max()+0.1)
    mae_theta = abs(df['Err_Theta']).mean()
    ax3.text(0.05, 0.9, f"MAE: {mae_theta:.3f} rad", transform=ax3.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    linearity_path = os.path.join(OUTPUT_DIR, "fig_lunarlander_linearity.pdf")
    plt.savefig(linearity_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {linearity_path}")
    
    print("\n=== Statistics for LaTeX ===")
    print(f"Calibrated X MAE: {mae_x:.4f}")
    print(f"Calibrated Y MAE: {mae_y:.4f}")
    print(f"Calibrated Theta MAE: {mae_theta:.4f}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[Fatal] Model file not found: {MODEL_PATH}")
    else:
        model = load_policy(MODEL_PATH)
        df_error = generate_lunarlander_calibrated_data(model)
        plot_lunarlander_analysis(df_error)