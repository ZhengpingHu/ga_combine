#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Configure device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

def collect_rotation_data(model_path: str, num_samples: int = 100):
    """
    Rotates the pendulum through 360 degrees while tracking the INFERRED position
    of the mathematically STATIC central pivot, using Absolute Distance Anchoring.
    """
    print(f"Loading YOLO model from {model_path} on {DEVICE}...")
    model = YOLO(model_path).to(DEVICE)
    
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset()
    
    # Pendulum-v1 的真实物理转轴永远固定在 500x500 画面的正中心
    gt_pivot_x, gt_pivot_y = 250.0, 250.0
    abs_center = np.array([gt_pivot_x, gt_pivot_y])
    
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    
    yolo_pivots_x = []
    yolo_pivots_y = []
    
    print(f"Collecting {num_samples} samples while rotating the pendulum...")
    for theta in angles:
        env.unwrapped.state = np.array([theta, 0.0])
        frame = env.render()
        
        results = model.predict(frame, verbose=False, device=DEVICE, conf=0.5)
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            # 确保至少检测到了 2 个点
            if kpts.shape[0] >= 2:
                pt0 = kpts[0]
                pt1 = kpts[1]
                
                # ==========================================
                # [终极逻辑] 绝对距离锚定：谁离 (250,250) 近，谁就是转轴！
                # ==========================================
                dist_pt0 = np.linalg.norm(pt0 - abs_center)
                dist_pt1 = np.linalg.norm(pt1 - abs_center)
                
                if dist_pt0 < dist_pt1:
                    pivot = pt0  # pt0 更近
                    tip = pt1
                else:
                    pivot = pt1  # pt1 更近
                    tip = pt0
                
                # 记录过滤后的“真·转轴”坐标
                yolo_pivots_x.append(pivot[0])
                yolo_pivots_y.append(pivot[1])
                
    env.close()
    return (gt_pivot_x, gt_pivot_y), np.array(yolo_pivots_x), np.array(yolo_pivots_y)

def plot_zero_centered_drift(gt_pivot, x_coords, y_coords, save_path="pivot_drift_zero_centered.pdf"):
    """
    Plots the Relative Pixel Error. Ground Truth is translated to (0, 0).
    """
    if len(x_coords) == 0:
        print("No keypoints detected. Check model and confidence threshold.")
        return

    # ==========================================
    # CORE MATH: Translate coordinates to (0,0)
    # ==========================================
    rel_x = x_coords - gt_pivot[0]
    rel_y = y_coords - gt_pivot[1]

    # Set up the plot style
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # 1. GROUND TRUTH: Translated to (0,0), 100% Red, Large Star, zorder=2 (on top of grid)
    ax.scatter([0.0], [0.0], 
               color='red', alpha=1.0, s=400, marker='*', edgecolor='black',
               label='True Physical Pivot (0, 0)', zorder=3)
    
    # 2. YOLO PREDICTIONS: 15% Opacity, zorder=2
    ax.scatter(rel_x, rel_y, 
               color='blue', alpha=0.15, s=100, edgecolor='none',
               label='YOLO Inferred Pivot', zorder=2)
    
    # ==========================================
    # DYNAMIC ZOOM-IN LOGIC
    # ==========================================
    # Find the maximum absolute deviation to create a perfectly symmetric box
    max_dev_x = np.max(np.abs(rel_x))
    max_dev_y = np.max(np.abs(rel_y))
    max_dev = max(max_dev_x, max_dev_y)
    
    # Add a tiny padding (e.g., 1.5 pixels) so points don't touch the edge
    plot_limit = np.ceil(max_dev) + 1.5 
    
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    
    # Formatting
    ax.set_title("Visual Noise Barrier: Spatial Drift of Static Pivot\n(Relative Pixel Error)", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Pixel Error (X-axis)", fontsize=13)
    ax.set_ylabel("Pixel Error (Y-axis)", fontsize=13)
    
    # Draw thicker crosshairs at x=0 and y=0 for reference
    ax.axhline(0, color='black', linewidth=1.2, alpha=0.8, zorder=1)
    ax.axvline(0, color='black', linewidth=1.2, alpha=0.8, zorder=1)
    
    # Add fine grid
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Fix the legend alpha so the blue dot is visible in the legend box
    leg = ax.legend(loc='upper right', framealpha=0.9)
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
        
    # Ensure 1 pixel on X equals 1 pixel on Y visually
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    MODEL_WEIGHTS = "best.pt" 
    
    if os.path.exists(MODEL_WEIGHTS):
        gt_pivot, x_data, y_data = collect_rotation_data(MODEL_WEIGHTS, num_samples=100)
        plot_zero_centered_drift(gt_pivot, x_data, y_data)
    else:
        print(f"Error: Could not find YOLO weights at {MODEL_WEIGHTS}")