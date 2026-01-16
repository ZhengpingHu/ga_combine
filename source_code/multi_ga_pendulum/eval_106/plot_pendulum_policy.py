# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.collections as mcoll

# ================= Config =================
MODEL_PATH = "./best_model_101.npz"
OUTPUT_DIR = "policy_visualization_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEEE Style
try: plt.style.use('seaborn-v0_8-paper')
except: plt.style.use('seaborn-paper')
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "axes.labelsize": 10, "font.size": 10, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": True # Show time axis
})

# Colors
COLOR_LEFT = "#D62728"  # Red (Negative Torque)
COLOR_RIGHT = "#1F77B4" # Blue (Positive Torque)

# ================= Internal Physics Simulator =================
# Simple Pendulum Dynamics (g=10, l=1, m=1)
def step_physics(th, thdot, u, dt=0.05):
    g = 10.0
    l = 1.0
    m = 1.0
    u = np.clip(u, -2.0, 2.0)
    # Dynamics equation
    newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -8, 8) # Max speed clip
    return newth, newthdot

# ================= Model =================
class FixedNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [cos, sin, dot] -> 3 dims
        self.net = nn.Linear(3, 1, bias=True) 
    def forward(self, x): return self.net(x)

def load_model(path):
    if not os.path.exists(path): return None
    data = np.load(path)
    model = FixedNNPolicy()
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(data['weights'][offset:offset+n]).view_as(p))
            offset += n
    model.eval()
    return model

# ================= Plotting =================
def plot_swingup_sequence(model):
    # 1. Run Simulation
    th = np.pi # Start DOWN
    thdot = 0.0
    dt = 0.05
    max_steps = 100 # 5 seconds
    
    trajectory = [] # List of (x_top, y_top, action_color)
    
    print("Simulating swing-up...")
    for t in range(max_steps):
        # Obs: [cos, sin, dot]
        # Note: Pendulum Env usually has 0 as UP. So cos(pi)=-1 (down).
        # We adjust to match standard definitions.
        obs = torch.tensor([[np.cos(th), np.sin(th), thdot]], dtype=torch.float32)
        
        with torch.no_grad():
            # Action is continuous
            u = torch.tanh(model(obs)).item() * 2.0
            
        # Store visualization data
        # Map torque to color
        # Normalize u (-2 to 2) -> 0 to 1 for colormap? Or just binary for clarity?
        # Let's use Gradient: Red <-> White <-> Blue
        norm_u = (u + 2) / 4.0 # 0.0 (Red) to 1.0 (Blue)
        cmap = plt.get_cmap('RdBu')
        color = cmap(norm_u)
        
        # Pendulum Tip Position
        # x = l * sin(th), y = -l * cos(th) (0 is Up)
        tip_x = np.sin(th)
        tip_y = np.cos(th) # 0 is Up, -1 is Down
        
        # We plot on a timeline. X = Time + offset
        trajectory.append((t, tip_x, tip_y, color, u))
        
        # Step
        th, thdot = step_physics(th, thdot, u, dt)
        
        # Stop if stabilized up
        if t > 20 and abs(th) < 0.1 and abs(thdot) < 0.1:
            break

    # 2. Plotting (Stroboscopic View)
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=300)
    
    # Subsample frames to avoid clutter
    step_size = 3 
    frames = trajectory[::step_size]
    
    # Plot baseline
    ax.axhline(0, color='black', lw=2) # The pivot point line
    
    for i, (t, x, y, c, u) in enumerate(frames):
        # Shift X by time to spread them out
        pivot_x = i * 1.5
        pivot_y = 0
        
        tip_visual_x = pivot_x + x
        tip_visual_y = pivot_y + y
        
        # Draw Rod
        ax.plot([pivot_x, tip_visual_x], [pivot_y, tip_visual_y], color='black', lw=2, alpha=0.3, zorder=1)
        
        # Draw Tip (Color = Action)
        ax.scatter(tip_visual_x, tip_visual_y, color=c, s=80, edgecolors='black', zorder=2)
        
        # Draw small torque arrow at pivot? (Optional, maybe too messy)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LEFT, markersize=10, label='Torque Left (CW)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#DDDDDD', markersize=10, label='Torque ~0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_RIGHT, markersize=10, label='Torque Right (CCW)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("Time Evolution (Left to Right)", fontweight='bold')
    
    # Add Text: "Start" and "Stabilized"
    ax.text(0, -1.3, "Start (Down)", ha='center', fontsize=9)
    ax.text((len(frames)-1)*1.5, 1.3, "Goal (Up)", ha='center', fontsize=9)

    out = os.path.join(OUTPUT_DIR, "fig_pendulum_trajectory.pdf")
    plt.savefig(out, format='pdf', bbox_inches='tight')
    print(f"Saved: {out}")

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    if model: plot_swingup_sequence(model)