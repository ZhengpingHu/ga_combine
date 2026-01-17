# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ================= 配置 =================
# 确保这里指向你的 Env 训练出的最佳模型
MODEL_PATH = "./best_model_final_101.npz" 
OUTPUT_DIR = "summary_analysis_baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 分辨率 (越高越细腻)
RESOLUTION = 400 

# IEEE 样式
try: plt.style.use('seaborn-v0_8-paper')
except: plt.style.use('seaborn-paper')
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "axes.labelsize": 10, "font.size": 10, "axes.titlesize": 10,
    "figure.constrained_layout.use": True
})

# === Env 专属配色方案 (Purple-White-Green) ===
# 负力矩(Left) -> 0 -> 正力矩(Right)
# Purple (#7b3294) -> White -> Green (#008837)
# 这是一个经典的 Diverging Colormap (PRGn 的变体)
nodes = [0.0, 0.5, 1.0]
colors = ["#7b3294", "#f7f7f7", "#008837"] 
ENV_CMAP = mcolors.LinearSegmentedColormap.from_list("EnvDiverging", list(zip(nodes, colors)))

# ================= 模型定义 (必须与训练时一致) =================
class PendulumEnvPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Env Agent 通常是: Input(3) -> Hidden(16) -> ReLU -> Output(1) -> Tanh
        # 如果你的结构不同，请务必修改这里！
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
    def forward(self, x): 
        # 输出范围 [-2, 2]
        return self.net(x) * 2.0

def load_model(path):
    if not os.path.exists(path): return None
    print(f"Loading model from {path}...")
    data = np.load(path)
    model = PendulumEnvPolicy()
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(data['weights'][offset:offset+n]).view_as(p))
            offset += n
    model.eval()
    return model

def get_state_input(theta, thdot):
    # Pendulum Observation: [cos(theta), sin(theta), theta_dot]
    # theta=0 是垂直向上
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    # 裁剪速度范围以免超出分布
    thdot_clipped = np.clip(thdot, -8, 8)
    return torch.tensor([[cos_th, sin_th, thdot_clipped]], dtype=torch.float32)

# ================= 绘图逻辑 =================
def plot_env_phase_heatmap(model):
    print("Generating Phase Space Heatmap...")
    
    # 1. 创建网格
    thetas = np.linspace(-np.pi, np.pi, RESOLUTION)
    thdots = np.linspace(-8, 8, RESOLUTION)
    T, TD = np.meshgrid(thetas, thdots)
    Actions = np.zeros_like(T)

    # 2. 批量推理 (或者逐点推理)
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            state = get_state_input(T[i, j], TD[i, j])
            with torch.no_grad():
                action = model(state).item()
            Actions[i, j] = action

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(4.0, 3.5), dpi=300)
    
    # 绘制热力图
    # vmin=-2, vmax=2 对应 Pendulum 的最大力矩
    c = ax.pcolormesh(T, TD, Actions, cmap=ENV_CMAP, vmin=-2, vmax=2, shading='auto', rasterized=True)
    
    # Colorbar
    cbar = fig.colorbar(c, ax=ax, pad=0.03)
    cbar.set_label('Applied Torque (N·m)', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)
    
    # 轴标签
    ax.set_xlabel(r"Angle ($\theta$)", fontweight='bold')
    ax.set_ylabel(r"Angular Velocity ($\dot{\theta}$)", fontweight='bold')
    
    # 设置刻度 (Pi)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-8, 8)
    ticks = [-np.pi, 0, np.pi]
    labels = [r'$-\pi$ (Down)', r'$0$ (Up)', r'$\pi$ (Down)']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    
    # 标注目标状态 (0, 0)
    ax.plot(0, 0, marker='*', color='gold', markersize=12, markeredgecolor='black', label='Goal State', zorder=10)
    
    # 标题 (可选，注明是基准)
    # ax.set_title("Policy Landscape (State-Based Baseline)", fontsize=10)

    # Legend
    ax.legend(loc='upper right', frameon=False, handletextpad=0.2)

    # 保存
    out_pdf = os.path.join(OUTPUT_DIR, "fig_env_policy_heatmap.pdf")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved: {out_pdf}")

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    if model: 
        plot_env_phase_heatmap(model)
    else:
        print(f"[Error] Model not found at {MODEL_PATH}")
        print("Please check the path or train an env-based model first.")