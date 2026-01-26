# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os

# ================= Config =================
MODEL_PATH = "./best_model_full_eval_101.npz"
OUTPUT_DIR = "policy_visualization_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEEE Style
try: plt.style.use('seaborn-v0_8-paper')
except: plt.style.use('seaborn-paper')
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "axes.labelsize": 10, "font.size": 10, 
    "axes.grid": False, # Grid can clutter vector fields
    "axes.spines.top": True, "axes.spines.right": True
})

# Colors
COLOR_PUSH_LEFT = "#D62728"   # Red (Force < 0)
COLOR_PUSH_RIGHT = "#1F77B4"  # Blue (Force > 0)

# ================= Internal Physics Simulator =================
def get_dynamics(position, velocity, force):
    """
    Returns the derivatives (v_dot, a_dot) for quiver/streamplot
    """
    force = np.clip(force, -1.0, 1.0)
    power = 0.0015
    force_scaled = force * power
    
    # Acceleration = Force - Gravity
    # v_next = v + a
    # p_next = p + v
    # We want instantaneous rate of change for vectors
    acceleration = force_scaled - 0.0025 * np.cos(3 * position)
    
    return velocity, acceleration # (dx/dt, dv/dt)

# ================= Model Definition =================
class EvolvedController(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

def load_model(path):
    if not os.path.exists(path): return EvolvedController()
    data = np.load(path)
    model = EvolvedController()
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            if offset + n <= len(data['weights']):
                p.data.copy_(torch.from_numpy(data['weights'][offset:offset+n]).view_as(p))
                offset += n
    model.eval()
    return model

# ================= Plotting: Global Streamplot =================
def plot_global_policy(model):
    # 1. Setup Grid
    # MountainCar ranges: x in [-1.2, 0.6], v in [-0.07, 0.07]
    res = 200 # Resolution for background
    x = np.linspace(-1.2, 0.6, res)
    v = np.linspace(-0.07, 0.07, res)
    XX, VV = np.meshgrid(x, v)
    
    # 2. Compute Policy Actions (Background Color)
    grid_tensor = torch.tensor(np.stack([XX.ravel(), VV.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        actions = model(grid_tensor).numpy().reshape(XX.shape)

    # 3. Compute Vector Field (Dynamics under Policy)
    # We need a coarser grid for arrows/streamlines to avoid clutter
    res_vec = 20 
    x_vec = np.linspace(-1.2, 0.6, res_vec)
    v_vec = np.linspace(-0.07, 0.07, res_vec)
    XX_vec, VV_vec = np.meshgrid(x_vec, v_vec)
    
    # Calculate action at these coarse points
    vec_tensor = torch.tensor(np.stack([XX_vec.ravel(), VV_vec.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        act_vec = model(vec_tensor).numpy().reshape(XX_vec.shape)
    
    # Calculate dx/dt (velocity) and dv/dt (acceleration)
    DX, DV = get_dynamics(XX_vec, VV_vec, act_vec)

    # 4. Visualization
    fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)

    # A. Background Contour (Policy)
    # Use binary or divergent map to show Push Left vs Push Right
    # Alpha reduced to make arrows visible
    cf = ax.contourf(XX, VV, actions, levels=50, cmap='RdBu', alpha=0.6)
    
    # B. Streamlines (The "Flow" of the system)
    strm = ax.streamplot(x_vec, v_vec, DX, DV, color='k', linewidth=0.6, density=1.2, arrowsize=0.8)
    
    # C. Overlay ONE explicit trajectory (The "Golden Path")
    pos, vel = -0.5, 0.0
    traj_x, traj_v = [pos], [vel]
    for _ in range(500):
        with torch.no_grad():
            act = model(torch.tensor([[pos, vel]], dtype=torch.float32)).item()
        # Simple dynamics step
        vel += (act * 0.0015) - 0.0025 * np.cos(3 * pos)
        vel = np.clip(vel, -0.07, 0.07)
        pos += vel
        pos = np.clip(pos, -1.2, 0.6)
        traj_x.append(pos)
        traj_v.append(vel)
        if pos >= 0.45: break
            
    # --- 修复点：正确使用 path_effects ---
    ax.plot(traj_x, traj_v, color='gold', linewidth=2.0, label='Successful Run', 
            path_effects=[path_effects.withStroke(linewidth=3, foreground="black")])
    
    # D. Annotations
    # --- 修复点：使用 raw string (r"...") 避免 SyntaxWarning ---
    ax.set_xlabel(r"Position ($x$)", fontweight='bold')
    ax.set_ylabel(r"Velocity ($\dot{x}$)", fontweight='bold')
    
    # Add Goal Line
    ax.axvline(0.45, color='#333333', linestyle='--', linewidth=1)
    ax.text(0.46, 0.05, 'Goal', rotation=90, fontsize=8)

    # Custom Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLOR_PUSH_LEFT, alpha=0.6, label='Action: Left'),
        Patch(facecolor=COLOR_PUSH_RIGHT, alpha=0.6, label='Action: Right'),
        Line2D([0], [0], color='k', lw=0.6, label='System Flow'),
        Line2D([0], [0], color='gold', lw=2.0, label='Actual Trajectory')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)
    
    out_path = os.path.join(OUTPUT_DIR, "mountaincar_global_flow.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    # Create dummy model if missing
    if not os.path.exists(MODEL_PATH):
        dummy = EvolvedController()
        all_weights = np.concatenate([p.data.numpy().flatten() for p in dummy.parameters()])
        np.savez(MODEL_PATH, weights=all_weights)

    model = load_model(MODEL_PATH)
    plot_global_policy(model)