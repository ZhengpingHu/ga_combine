# -*- coding: utf-8 -*-
# HSR Heatmap Analysis Tool (IEEE CEC Edition)

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
import matplotlib.tri as tri

# ================= 配置区域 =================
MODEL_PATH = "best.pt"       # 你的模型路径
ANGLE_GAIN = 1.12            # 必须与服务器代码一致！
SAMPLE_COUNT = 2000          # 增加采样点以获得更平滑的热力图
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 采样范围
X_RANGE = (-2.2, 2.2)        
THETA_RANGE = (-0.25, 0.25)  

# ================= 0. IEEE 样式配置 =================
OUTPUT_DIR = "analysis_plots_ieee"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')

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

# 定义论文统一配色
COLOR_SCATTER = "#1f77b4"  # Royal Blue (数据点)
COLOR_IDEAL   = "black"    # 理想线
COLOR_ERROR   = "#d62728"  # Brick Red (误差)

# ================= 1. 核心估算器 (保持不变) =================
class LocalEstimator:
    def __init__(self, model_path, gain):
        self.device = torch.device(DEVICE)
        print(f"[Analysis] Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path).to(self.device)
        self.gain = gain
        self.screen_width = 600
        self.world_width = 4.8
        self.scale = self.screen_width / self.world_width

    @torch.no_grad()
    def infer_single_frame(self, frame_rgb):
        frame_bgr = frame_rgb[..., ::-1].copy()
        results = self.model.predict(frame_bgr, verbose=False, device=self.device)
        if not results or len(results) == 0: return None
        r = results[0]
        if r.keypoints is None or r.keypoints.has_visible is False: return None
        kpts = r.keypoints.xy[0].cpu().numpy()
        if len(kpts) < 2: return None
        
        cx, cy = kpts[0]
        tx, ty = kpts[1]

        raw_x = (cx - (self.screen_width / 2.0)) / self.scale
        dx = tx - cx
        dy = ty - cy
        base_angle = math.atan2(dx, -dy)
        raw_theta = base_angle * self.gain
        return raw_x, raw_theta

# ================= 2. 数据生成 (保持不变) =================
def generate_heatmap_data():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} not found.")
        return pd.DataFrame()

    estimator = LocalEstimator(MODEL_PATH, ANGLE_GAIN)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()
    data_log = []

    print(f"Starting sampling loop ({SAMPLE_COUNT} samples)...")
    for i in tqdm(range(SAMPLE_COUNT)):
        rand_x = np.random.uniform(X_RANGE[0], X_RANGE[1])
        rand_theta = np.random.uniform(THETA_RANGE[0], THETA_RANGE[1])
        
        state_vector = np.array([rand_x, 0.0, rand_theta, 0.0])
        env.unwrapped.state = state_vector
        frame = env.render()
        res = estimator.infer_single_frame(frame)
        
        if res is not None:
            pred_x, pred_theta = res
            data_log.append({
                "True_X": rand_x,
                "True_Theta": rand_theta,
                "Pred_X": pred_x,
                "Pred_Theta": pred_theta,
                "Abs_Error_X": abs(pred_x - rand_x),
                "Abs_Error_Theta": abs(pred_theta - rand_theta),
                "Error_Theta_Signed": pred_theta - rand_theta
            })

    env.close()
    return pd.DataFrame(data_log)

# ================= 3. 绘图逻辑 (IEEE CEC 核心修改) =================
def plot_heatmaps_ieee(df):
    if len(df) == 0: return
    print("\nGenerating IEEE Style Plots...")

    # --- 图 A: 空间误差分布 (Spatial Error Distribution) ---
    # 目的：展示误差是否随位置变化（例如镜头畸变导致边缘误差大）
    fig, ax = plt.subplots(figsize=(4.0, 3.5), dpi=300)
    
    # 准备三角剖分插值
    triang = tri.Triangulation(df["True_X"], df["True_Theta"])
    
    # 使用 'Reds' 色板：白色=完美，红色=误差大。非常直观。
    # vmin=0 确保白色是0误差
    contour = ax.tricontourf(triang, df["Abs_Error_Theta"], levels=20, cmap="Reds", vmin=0)
    
    # 添加物理限制框 (CartPole Limit)
    ax.axvline(-2.4, color='black', ls=':', lw=1, alpha=0.5)
    ax.axvline(2.4, color='black', ls=':', lw=1, alpha=0.5)
    
    # Colorbar
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Angle Estimation Error (rad)', rotation=270, labelpad=15)
    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    ax.set_title("Spatial Error Distribution", fontweight='bold')
    ax.set_xlabel("Cart Position (m)")
    ax.set_ylabel("Pole Angle (rad)")
    ax.set_xlim(X_RANGE)
    ax.set_ylim(THETA_RANGE)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_spatial_error_heatmap.pdf"), format='pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/fig_spatial_error_heatmap.pdf")

    # --- 图 B: 线性度验证 (Linearity Verification) ---
    # 目的：展示 HSR 的线性拟合能力。左图是拟合效果，右图是残差。
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=300)

    # 子图 1: Predicted vs True
    ax1.scatter(df["True_Theta"], df["Pred_Theta"], s=3, color=COLOR_SCATTER, alpha=0.5, rasterized=True)
    # 画对角线 y=x
    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    ]
    ax1.plot(lims, lims, color=COLOR_IDEAL, ls='--', lw=1, label="Ideal")
    
    ax1.set_xlabel("True Angle (rad)")
    ax1.set_ylabel("Predicted Angle (rad)")
    ax1.set_title("HSR Linearity", fontsize=10)
    ax1.legend(frameon=False, loc='upper left')

    # 子图 2: Residuals (Error vs True)
    ax2.scatter(df["True_Theta"], df["Error_Theta_Signed"], s=3, color=COLOR_ERROR, alpha=0.5, rasterized=True)
    ax2.axhline(0, color=COLOR_IDEAL, ls='--', lw=1)
    
    ax2.set_xlabel("True Angle (rad)")
    ax2.set_ylabel("Residual Error (rad)")
    ax2.set_title("Residual Analysis", fontsize=10)
    
    # 标注 MAE
    mae = df['Abs_Error_Theta'].mean()
    ax2.text(0.05, 0.9, f"MAE = {mae:.4f} rad", transform=ax2.transAxes, 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_hsr_linearity.pdf"), format='pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/fig_hsr_linearity.pdf")

    # --- 统计输出 ---
    print("\n=== Error Statistics (LaTeX) ===")
    print(f"MAE Angle: {df['Abs_Error_Theta'].mean():.4f}")
    print(f"Max Angle: {df['Abs_Error_Theta'].max():.4f}")
    print(f"MAE Pos:   {df['Abs_Error_X'].mean():.4f}")

if __name__ == "__main__":
    df = generate_heatmap_data()
    plot_heatmaps_ieee(df)