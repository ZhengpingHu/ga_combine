#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LunarLander Vision System Analysis Tool (Open-loop Free-fall Edition)
# Focus: Absolute Spatial Perception Error Measurement

import os
# ================= CRITICAL FIX =================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# ================================================

import numpy as np
import pandas as pd
import gymnasium as gym
from multiprocessing.connection import Client
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from tqdm import tqdm

# ================= Configuration =================
SAMPLE_COUNT = 3000      # 采样点数（自由落体轨迹比较短，增加点数可以让热力图更平滑）
MAX_STEPS = 1000         # 防止单局卡死的最大步数

# ================= IEEE Styling =================
OUTPUT_DIR = "analysis_plots_lunarlander"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try: 
    plt.style.use('seaborn-v0_8-paper')
except: 
    plt.style.use('seaborn-paper')

plt.rcParams.update({
    "font.family": "serif", 
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 11, 
    "font.size": 11, 
    "legend.fontsize": 9,
    "xtick.labelsize": 10, 
    "ytick.labelsize": 10,
    "axes.grid": True, 
    "grid.alpha": 0.3, 
    "grid.linestyle": "--",
    "axes.spines.top": False, 
    "axes.spines.right": False,
    "figure.constrained_layout.use": True, 
    "axes.unicode_minus": False
})

# ================= RPC Definitions =================
class RPCClient:
    def __init__(self, host="127.0.0.1", port=6001, authkey="yolo-rpc"):
        self.address = (host, port)
        self.authkey = authkey.encode("utf-8")
        self.conn = None
        
    def __enter__(self):
        try:
            self.conn = Client(self.address, authkey=self.authkey)
            return self
        except Exception as e: 
            print(f"[!] RPC Connection failed: {e}")
            return None
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()
        
    def reset(self):
        if self.conn:
            try: 
                self.conn.send(("reset", None))
                _ = self.conn.recv()
            except: pass
            
    def update_full(self, frame_bgr, steps=0):
        if self.conn:
            try:
                self.conn.send(("update_full", (frame_bgr, steps)))
                ok, z = self.conn.recv()
                return z if ok else None
            except: 
                return None
        return None

# ================= Data Sampling & Auto-Calibration =================
def collect_freefall_calibrated_data():
    print(f"[*] 准备收集 {SAMPLE_COUNT} 个自由落体数据点 (Open-loop Vision Testing)...")
    raw_data = []
    
    # 使用无控制器的纯净环境
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    pbar = tqdm(total=SAMPLE_COUNT, desc="Sampling")
    collected = 0
    seed = 101 # 设定起始种子，保证每次运行收集到的轨迹一致
    
    try:
        rpc = RPCClient()
        with rpc:
            if rpc.conn is None:
                raise RuntimeError("无法连接到 YOLO RPC 服务端，请确认服务端已启动！")
                
            while collected < SAMPLE_COUNT:
                obs, _ = env.reset(seed=seed)
                seed += 1
                rpc.reset()
                last_valid_state = np.zeros(3, dtype=np.float32)
                
                for t in range(MAX_STEPS):
                    if collected >= SAMPLE_COUNT: 
                        break
                    
                    frame = env.render()
                    if frame is None: 
                        break
                    
                    frame_bgr = frame[..., ::-1].copy()
                    pred_state = rpc.update_full(frame_bgr, t)
                    
                    if pred_state is None: 
                        pred_state = last_valid_state
                    else: 
                        last_valid_state = pred_state
                        
                    # Ground Truth
                    true_x = obs[0]
                    true_y = obs[1]
                    true_theta = obs[4]
                    
                    # YOLO Vision Prediction
                    raw_pred_x = pred_state[0]
                    raw_pred_y = pred_state[1]
                    raw_pred_theta = pred_state[2]
                    
                    raw_data.append({
                        "True_X": true_x, "True_Y": true_y, "True_Theta": true_theta,
                        "Raw_Pred_X": raw_pred_x, "Raw_Pred_Y": raw_pred_y, "Raw_Pred_Theta": raw_pred_theta
                    })
                    collected += 1
                    pbar.update(1)
                    
                    # 【核心修改】：强制 action = 0 (Do Nothing / Free-fall)
                    obs, reward, done, truncated, info = env.step(0) 
                    
                    # 如果航天器坠毁或飞出边界，结束这一局，重置种子继续下一局
                    if done or truncated: 
                        break
                    
    except Exception as e:
        print(f"\n[Fatal Error] 数据收集异常中断: {e}")
        
    pbar.close()
    env.close()
    
    df = pd.DataFrame(raw_data)
    
    print("\n[*] 正在执行 Auto-Calibration (一维线性拟合映射)...")
    
    # 拟合线性模型，将 YOLO 的像素/相对坐标 映射到 Gym 的物理坐标：True = m * Raw_Pred + c
    coef_x = np.polyfit(df['Raw_Pred_X'], df['True_X'], 1)
    coef_y = np.polyfit(df['Raw_Pred_Y'], df['True_Y'], 1)
    coef_theta = np.polyfit(df['Raw_Pred_Theta'], df['True_Theta'], 1)
    
    df['Calib_Pred_X'] = df['Raw_Pred_X'] * coef_x[0] + coef_x[1]
    df['Calib_Pred_Y'] = df['Raw_Pred_Y'] * coef_y[0] + coef_y[1]
    df['Calib_Pred_Theta'] = df['Raw_Pred_Theta'] * coef_theta[0] + coef_theta[1]
    
    # 计算误差
    df['Err_X'] = df['Calib_Pred_X'] - df['True_X']
    df['Err_Y'] = df['Calib_Pred_Y'] - df['True_Y']
    
    # 角度误差计算（考虑周期性翻转问题）
    df['Err_Theta'] = np.arctan2(np.sin(df['Calib_Pred_Theta'] - df['True_Theta']), 
                                 np.cos(df['Calib_Pred_Theta'] - df['True_Theta']))
    
    # 计算综合空间绝对欧氏误差 (Total Euclidean Spatial Error)
    df['Total_Pos_Err'] = np.sqrt(df['Err_X']**2 + df['Err_Y']**2)
    
    return df

# ================= Plotting =================
def plot_lunarlander_spatial_error(df):
    if len(df) == 0: 
        print("[!] 错误：数据为空，无法绘图。")
        return
        
    print("[*] 正在生成空间误差热力分布图...")

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    
    # 使用 Delaunay 三角剖分进行不规则散点的等高线渲染
    triang = tri.Triangulation(df["True_X"], df["True_Y"])
    
    # 使用 Reds 颜色映射，层数设为 25 让过渡更平滑
    cnt = ax.tricontourf(triang, df["Total_Pos_Err"], levels=25, cmap="Reds", vmin=0)
    
    # 添加颜色条
    cbar = fig.colorbar(cnt, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Calibrated Position Error', rotation=270, labelpad=15, fontweight='bold')
    
    ax.set_title("Calibrated Spatial Error Map (Free-fall Trajectories)", fontweight='bold', pad=10)
    ax.set_xlabel("True Horizontal Position (X)")
    ax.set_ylabel("True Vertical Position (Y)")
    
    # 添加原点十字参考线
    ax.axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
    
    # 根据数据动态设置边界，让“穹顶”居中
    x_min, x_max = df["True_X"].min(), df["True_X"].max()
    y_min, y_max = df["True_Y"].min(), df["True_Y"].max()
    ax.set_xlim(x_min - 0.05, x_max + 0.05)
    ax.set_ylim(y_min - 0.05, y_max + 0.05)
    
    spatial_png = os.path.join(OUTPUT_DIR, "fig_lunarlander_spatial_error.png")
    spatial_pdf = os.path.join(OUTPUT_DIR, "fig_lunarlander_spatial_error.pdf")
    plt.savefig(spatial_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(spatial_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] 空间误差图已保存至: {spatial_png}")

    print("\n=== 感知误差统计 (LaTeX 友好) ===")
    print(f"Mean Abs Error (X):     {abs(df['Err_X']).mean():.4f}")
    print(f"Mean Abs Error (Y):     {abs(df['Err_Y']).mean():.4f}")
    print(f"Mean Abs Error (Theta): {abs(df['Err_Theta']).mean():.4f} rad")
    print(f"Max Spatial Error:      {df['Total_Pos_Err'].max():.4f}")
    print("===================================")

if __name__ == "__main__":
    df_error = collect_freefall_calibrated_data()
    plot_lunarlander_spatial_error(df_error)