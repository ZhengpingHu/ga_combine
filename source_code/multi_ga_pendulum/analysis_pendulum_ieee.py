# -*- coding: utf-8 -*-
# Pendulum HSR Analysis Tool (IEEE CEC Edition)
# Matches strict IEEE formatting standards

import os
import math
import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from tqdm import tqdm

# ================= 配置区域 =================
MODEL_PATH = "best.pt"       # 你的 Pendulum YOLO 模型路径
IMG_SIZE = 640               # 与训练时一致
SAMPLE_COUNT = 2000          # 采样点数量
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 角度范围 (-pi 到 pi)
RANGE_THETA = (-np.pi, np.pi)

# ================= 0. IEEE 样式配置 =================
OUTPUT_DIR = "analysis_plots_pendulum"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try: plt.style.use('seaborn-v0_8-paper')
except: plt.style.use('seaborn-paper')

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "axes.labelsize": 10, "font.size": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True, "axes.unicode_minus": False
})

# 论文配色
COLOR_SCATTER = "#1f77b4"  # Royal Blue
COLOR_IDEAL   = "black"
COLOR_ERROR   = "#d62728"  # Brick Red

# ================= 1. 核心估算器 (逻辑复刻自 Server) =================
class PendulumLocalEstimator:
    def __init__(self, model_path, img_size):
        self.device = torch.device(DEVICE)
        print(f"[Analysis] Loading YOLO from {model_path}...")
        self.model = YOLO(model_path).to(self.device)
        self.img_size = img_size

    @torch.no_grad()
    def infer_and_recover_angle(self, frame_rgb):
        """
        推理并逆向解算角度
        """
        frame_bgr = frame_rgb[..., ::-1].copy()
        results = self.model.predict(frame_bgr, verbose=False, device=self.device, conf=0.5)
        
        # 1. 基础检查
        if not results or len(results) == 0: return None
        r = results[0]
        
        # 2. 关键点检查 (Robust Check)
        if r.keypoints is None or r.keypoints.xy is None: return None
        if len(r.keypoints.xy) == 0: return None
        if r.keypoints.xy.shape[1] < 2: return None
        
        kpts = r.keypoints.xy[0].cpu().numpy() # [Pivot, Tip]
        pivot_x, pivot_y = kpts[0]
        tip_x, tip_y = kpts[1]
        
        # --- 几何反解 (Geometric Recovery) ---
        # Server 逻辑: 
        # norm_x = (tip_x - pivot_x) / scale
        # norm_y = (tip_y - pivot_y) / scale
        
        scale = self.img_size / 2.0
        norm_x = (tip_x - pivot_x) / scale
        norm_y = (tip_y - pivot_y) / scale
        
        # Gym 坐标系定义:
        # Theta=0 是垂直向上 (Upright)
        # 屏幕坐标系 Y 向下。所以垂直向上对应屏幕的 -Y 方向。
        # atan2(x, -y):
        #   Tip在上方 (0, -1) -> atan2(0, 1) = 0 (符合)
        #   Tip在右方 (1, 0)  -> atan2(1, 0) = pi/2 (符合)
        #   Tip在下方 (0, 1)  -> atan2(0, -1) = pi (符合)
        
        theta_pred = math.atan2(norm_x, -norm_y)
        
        return theta_pred

# ================= 2. 数据采样 =================
def generate_pendulum_data():
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model {MODEL_PATH} not found.")
        return pd.DataFrame()

    estimator = PendulumLocalEstimator(MODEL_PATH, IMG_SIZE)
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset()
    
    data_log = []
    print(f"Sampling {SAMPLE_COUNT} points from phase space...")
    
    for _ in tqdm(range(SAMPLE_COUNT)):
        # 1. 随机角度 (-pi 到 pi)
        rand_theta = np.random.uniform(*RANGE_THETA)
        rand_speed = 0.0 # 静态测试
        
        env.unwrapped.state = np.array([rand_theta, rand_speed])
        
        # 2. 渲染 & 推理
        frame = env.render()
        res = estimator.infer_and_recover_angle(frame)
        
        if res is not None:
            pred_theta = res
            
            # 计算最小角度误差 (处理周期性跳变)
            # diff = atan2(sin(a-b), cos(a-b))
            diff = pred_theta - rand_theta
            angle_error = math.atan2(math.sin(diff), math.cos(diff))
            
            # 计算可视化的物理位置 (用于画圆环图)
            # 0度在正上方 (0, 1)
            true_x = math.sin(rand_theta)
            true_y = math.cos(rand_theta)
            
            data_log.append({
                "True_Theta": rand_theta,
                "Pred_Theta": pred_theta,
                "Error_Theta": angle_error,
                "Abs_Error": abs(angle_error),
                "Vis_X": true_x, # 用于散点图 X
                "Vis_Y": true_y  # 用于散点图 Y
            })
            
    env.close()
    return pd.DataFrame(data_log)

# ================= 3. 绘图 (IEEE Style) =================
def plot_pendulum_analysis(df):
    if len(df) == 0: return
    print("\nGenerating Pendulum Analysis Plots...")

    # --- 图 A: 环形误差分布 (Circular Error Map) ---
    # 我们画一个圆环，颜色代表误差大小。
    # 这能直观展示：在倒立点(0)、水平点(pi/2)、最低点(pi)的识别精度
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    
    # 使用散点图模拟圆环热力图
    sc = ax.scatter(df["Vis_X"], df["Vis_Y"], c=df["Abs_Error"], 
                    cmap="Reds", s=10, alpha=0.8, vmin=0, edgecolors='none')
    
    # 装饰
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # 画一个参考圆
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, ls='--', lw=0.5, alpha=0.3)
    ax.add_artist(circle)
    
    # 标注关键方位
    ax.text(0, 1.05, "Up (0)", ha='center', fontsize=8)
    ax.text(0, -1.15, "Down ($\pm\pi$)", ha='center', fontsize=8)
    ax.text(1.05, 0, "Right", va='center', fontsize=8)
    ax.text(-1.05, 0, "Left", va='center', fontsize=8)
    
    # 去掉刻度，只保留图形
    ax.axis('off')
    ax.set_title("Spatial Error Distribution", fontweight='bold', y=1.05)
    
    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Angle Error (rad)', rotation=270, labelpad=15)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_pendulum_spatial_error.pdf"), format='pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/fig_pendulum_spatial_error.pdf")

    # --- 图 B: 线性度验证 (Linearity) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=300)
    
    # Subplot 1: Predicted vs True
    ax1.scatter(df["True_Theta"], df["Pred_Theta"], s=3, c=COLOR_SCATTER, alpha=0.5, rasterized=True)
    ax1.plot([-4, 4], [-4, 4], c=COLOR_IDEAL, ls='--', lw=1)
    
    ax1.set_title("HSR Linearity")
    ax1.set_xlabel(r"True Angle ($\theta$)")
    ax1.set_ylabel(r"Predicted Angle ($\hat{\theta}$)")
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-np.pi, np.pi)
    
    ticks = [-np.pi, 0, np.pi]
    labels = [r'$-\pi$', '0', r'$\pi$']
    ax1.set_xticks(ticks); ax1.set_xticklabels(labels)
    ax1.set_yticks(ticks); ax1.set_yticklabels(labels)

    # Subplot 2: Residuals
    ax2.scatter(df["True_Theta"], df["Error_Theta"], s=3, c=COLOR_ERROR, alpha=0.5, rasterized=True)
    ax2.axhline(0, c=COLOR_IDEAL, ls='--', lw=1)
    
    ax2.set_title("Residual Analysis")
    ax2.set_xlabel(r"True Angle ($\theta$)")
    ax2.set_ylabel("Error (rad)")
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_xticks(ticks); ax2.set_xticklabels(labels)
    
    # Add MAE
    mae = df['Abs_Error'].mean()
    ax2.text(0.05, 0.9, f"MAE: {mae:.3f} rad", transform=ax2.transAxes, 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_pendulum_linearity.pdf"), format='pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/fig_pendulum_linearity.pdf")
    
    print("\n=== Statistics for LaTeX ===")
    print(f"MAE Angle: {mae:.4f}")
    print(f"Max Error: {df['Abs_Error'].max():.4f}")

if __name__ == "__main__":
    df = generate_pendulum_data()
    plot_pendulum_analysis(df)