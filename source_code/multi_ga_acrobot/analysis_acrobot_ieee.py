# -*- coding: utf-8 -*-
# Acrobot HSR Analysis Tool (IEEE CEC Edition)
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
import matplotlib.tri as tri

# ================= 配置区域 =================
MODEL_PATH = "./best.pt"       # 你的 Acrobot YOLO 模型路径
IMG_SIZE = 640               # 与训练时一致
SAMPLE_COUNT = 2000          # 采样点数量
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Acrobot 关节范围 (-pi 到 pi)
RANGE_THETA1 = (-np.pi, np.pi)
RANGE_THETA2 = (-np.pi, np.pi)

# ================= 0. IEEE 样式配置 =================
OUTPUT_DIR = "analysis_plots_acrobot"
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
class AcrobotLocalEstimator:
    def __init__(self, model_path, img_size):
        self.device = torch.device(DEVICE)
        print(f"[Analysis] Loading YOLO from {model_path}...")
        self.model = YOLO(model_path).to(self.device)
        self.img_size = img_size

    @torch.no_grad()
    def infer_and_recover_angles(self, frame_rgb):
        """
        推理并逆向解算角度 (用于验证精度)
        """
        frame_bgr = frame_rgb[..., ::-1].copy()
        results = self.model.predict(frame_bgr, verbose=False, device=self.device, conf=0.5)
        
        # 1. 基础结果检查
        if not results or len(results) == 0: 
            return None
        
        r = results[0]
        
        # 2. 关键点存在性检查 (CRITICAL FIX)
        # 必须检查: 
        # a) keypoints 对象是否存在
        # b) xy 属性是否存在
        # c) xy 的长度是否大于0 (是否有检测实例)
        if r.keypoints is None or r.keypoints.xy is None:
            return None
            
        if len(r.keypoints.xy) == 0: 
            return None # 修正了这里：如果检测列表为空，直接返回，不再访问 [0]

        # 3. 关键点数量检查 (Acrobot 需要 Base, Joint, Tip 共3个点)
        if r.keypoints.xy.shape[1] < 3: 
            return None
        
        # 4. 安全提取
        kpts = r.keypoints.xy[0].cpu().numpy() # [Base, Joint, Tip]
        base_x, base_y = kpts[0]
        joint_x, joint_y = kpts[1]
        tip_x, tip_y = kpts[2]
        
        # --- 几何反解 (Geometric Recovery) ---
        # Gym 坐标系: Y轴向下。
        
        # 计算 Theta 1 (Base -> Joint)
        dx1 = joint_x - base_x
        dy1 = joint_y - base_y
        theta1_pred = math.atan2(dx1, dy1)
        
        # 计算 Theta 2 (Joint -> Tip, 相对角度)
        dx2 = tip_x - joint_x
        dy2 = tip_y - joint_y
        theta_absolute_2 = math.atan2(dx2, dy2)
        
        # 相对角度 = 绝对角度2 - 绝对角度1
        theta2_pred = theta_absolute_2 - theta1_pred
        
        # 归一化到 [-pi, pi]
        theta2_pred = (theta2_pred + np.pi) % (2 * np.pi) - np.pi
        
        return theta1_pred, theta2_pred

# ================= 2. 数据采样 =================
def generate_acrobot_data():
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model {MODEL_PATH} not found.")
        return pd.DataFrame()

    estimator = AcrobotLocalEstimator(MODEL_PATH, IMG_SIZE)
    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    env.reset()
    
    data_log = []
    print(f"Sampling {SAMPLE_COUNT} points from state space...")
    
    for _ in tqdm(range(SAMPLE_COUNT)):
        # 1. 随机状态
        t1 = np.random.uniform(*RANGE_THETA1)
        t2 = np.random.uniform(*RANGE_THETA2)
        # 速度设为0，专注于静态识别精度
        env.unwrapped.state = np.array([t1, t2, 0.0, 0.0])
        
        # 2. 渲染 & 推理
        frame = env.render()
        res = estimator.infer_and_recover_angles(frame)
        
        if res is not None:
            p_t1, p_t2 = res
            
            # 计算误差 (考虑周期性)
            err1 = math.atan2(math.sin(p_t1 - t1), math.cos(p_t1 - t1))
            err2 = math.atan2(math.sin(p_t2 - t2), math.cos(p_t2 - t2))
            
            data_log.append({
                "True_T1": t1, "True_T2": t2,
                "Pred_T1": p_t1, "Pred_T2": p_t2,
                "Err_T1": err1, "Err_T2": err2,
                "Abs_Err_T1": abs(err1), "Abs_Err_T2": abs(err2),
                "Total_Err": abs(err1) + abs(err2) # 综合误差
            })
            
    env.close()
    return pd.DataFrame(data_log)

# ================= 3. 绘图 (IEEE Style) =================
def plot_acrobot_analysis(df):
    if len(df) == 0: return
    print("\nGenerating Acrobot Analysis Plots...")

    # --- 图 A: 状态空间误差热力图 (Configuration Space Error) ---
    # X轴: Theta 1, Y轴: Theta 2
    # 这能看出是否在某些姿态下(比如折叠在一起)容易识别错
    fig, ax = plt.subplots(figsize=(4.2, 3.5), dpi=300)
    
    triang = tri.Triangulation(df["True_T1"], df["True_T2"])
    # 使用 'Reds'，vmin=0
    cnt = ax.tricontourf(triang, df["Total_Err"], levels=20, cmap="Reds", vmin=0)
    
    cbar = fig.colorbar(cnt, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Total Angle Error (rad)', rotation=270, labelpad=15)
    
    ax.set_title("Configuration Space Error Map", fontweight='bold')
    ax.set_xlabel(r"Shoulder Angle ($\theta_1$)")
    ax.set_ylabel(r"Elbow Angle ($\theta_2$)")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    
    # 标注刻度为 pi
    ticks = [-np.pi, 0, np.pi]
    labels = [r'$-\pi$', '0', r'$\pi$']
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_acrobot_spatial_error.pdf"), format='pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/fig_acrobot_spatial_error.pdf")

    # --- 图 B: 线性度验证 (双关节) ---
    # 左图: Theta 1, 右图: Theta 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=300)
    
    # Joint 1
    ax1.scatter(df["True_T1"], df["Pred_T1"], s=2, c=COLOR_SCATTER, alpha=0.5, rasterized=True)
    ax1.plot([-4, 4], [-4, 4], c=COLOR_IDEAL, ls='--', lw=1)
    ax1.set_title(r"Shoulder ($\theta_1$) Linearity")
    ax1.set_xlabel("True Angle (rad)")
    ax1.set_ylabel("Predicted Angle (rad)")
    ax1.set_xticks(ticks); ax1.set_xticklabels(labels)
    ax1.set_yticks(ticks); ax1.set_yticklabels(labels)
    ax1.set_xlim(-np.pi, np.pi); ax1.set_ylim(-np.pi, np.pi)
    # Add MAE
    mae1 = df['Abs_Err_T1'].mean()
    ax1.text(0.05, 0.9, f"MAE: {mae1:.3f} rad", transform=ax1.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Joint 2
    ax2.scatter(df["True_T2"], df["Pred_T2"], s=2, c=COLOR_SCATTER, alpha=0.5, rasterized=True)
    ax2.plot([-4, 4], [-4, 4], c=COLOR_IDEAL, ls='--', lw=1)
    ax2.set_title(r"Elbow ($\theta_2$) Linearity")
    ax2.set_xlabel("True Angle (rad)")
    # ax2.set_ylabel("Predicted Angle") # 省略 Y 轴标签以节省空间
    ax2.set_xticks(ticks); ax2.set_xticklabels(labels)
    ax2.set_yticks(ticks); ax2.set_yticklabels(labels)
    ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(-np.pi, np.pi)
    # Add MAE
    mae2 = df['Abs_Err_T2'].mean()
    ax2.text(0.05, 0.9, f"MAE: {mae2:.3f} rad", transform=ax2.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_acrobot_linearity.pdf"), format='pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/fig_acrobot_linearity.pdf")
    
    # 打印 LaTeX 统计
    print("\n=== Statistics for LaTeX ===")
    print(f"Theta1 MAE: {mae1:.4f}")
    print(f"Theta2 MAE: {mae2:.4f}")

if __name__ == "__main__":
    df = generate_acrobot_data()
    plot_acrobot_analysis(df)