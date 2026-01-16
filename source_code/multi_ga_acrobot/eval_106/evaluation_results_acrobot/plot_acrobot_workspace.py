# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ================= Config =================
MODEL_PATH = "./best_model_101.npz" 
OUTPUT_DIR = "policy_visualization_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESOLUTION = 150 # Grid density (Higher = smoother)

# IEEE Styling
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 10, "axes.labelsize": 10, "axes.grid": False,
    "figure.constrained_layout.use": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False # Clean look
})

# Colors
COLOR_LEFT = "#D62728"  # Red
COLOR_RIGHT = "#1F77B4" # Blue
COLOR_NOOP = "#EEEEEE"  # Grey

# ================= Model & Physics =================
class FixedNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 3, bias=True) # Assuming 8 inputs
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

def get_8d_input_and_tip_pos(theta1, theta2):
    """
    Returns:
    1. tensor_input: 1x8 tensor for the model
    2. tip_x, tip_y: Physical coordinates of the end-effector
    """
    l1, l2 = 1.0, 1.0 # Standard Acrobot lengths
    
    # Forward Kinematics (Tip Position)
    # Acrobot: 0 angle is hanging DOWN in standard physics, 
    # but check your specific Env! Assuming standard Gym: 
    # y is down? No, standard math: y up.
    # Let's use standard trigonometric projection.
    
    # Joint 1
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1) # 0 is hanging down
    
    # Joint 2 (Tip)
    tip_x = x1 + l2 * np.sin(theta1 + theta2)
    tip_y = y1 - l2 * np.cos(theta1 + theta2)
    
    # Construct 8D input (Approximate for visualization)
    # Assuming input is [cos1, sin1, cos2, sin2, d1, d2, ...] OR raw angles
    # Using specific 8D format based on typical HSR: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    # Velocities set to 0 to see "Static Policy"
    inp = torch.tensor([[x1, y1, tip_x, tip_y, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    
    return inp, tip_x, tip_y

# ================= Plotting =================
def plot_acrobot_workspace(model):
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300)

    # 1. Generate Point Cloud in Joint Space
    theta1 = np.linspace(-np.pi, np.pi, RESOLUTION)
    theta2 = np.linspace(-np.pi, np.pi, RESOLUTION)
    
    # Lists to store plotting data
    X_points, Y_points, Colors = [], [], []
    
    print("Mapping workspace...")
    for t1 in theta1:
        for t2 in theta2:
            inp, x, y = get_8d_input_and_tip_pos(t1, t2)
            
            with torch.no_grad():
                logits = model(inp)
                action = torch.argmax(logits).item()
            
            # Map action to color
            if action == 0: c = COLOR_LEFT   # Torque -1
            elif action == 1: c = COLOR_NOOP # Torque 0
            else: c = COLOR_RIGHT            # Torque +1
            
            X_points.append(x)
            Y_points.append(y)
            Colors.append(c)

    # 2. Scatter Plot (Pixels)
    # Using scatter is inefficient for dense grids, but handles the non-linear mapping 
    # from theta->(x,y) better than pcolormesh
    ax.scatter(X_points, Y_points, c=Colors, s=1.5, marker='s', edgecolors='none', alpha=0.6)

    # 3. Add Reference Lines
    # Draw the workspace limit (Max reach = 2.0)
    circle = plt.Circle((0, 0), 2.0, color='black', fill=False, linestyle='--', alpha=0.3)
    ax.add_artist(circle)
    # Draw inner limit (Min reach = 0 if lengths equal)
    
    # 4. Simulate a Trajectory (Optional visual aid)
    # Just drawing a representative "Swing Up" curve
    # Start bottom (0,-2) -> Swing Right -> Swing Left -> Top (0,2)
    traj_x = [0, 1.2, 0.8, -1.5, -0.5, 0]
    traj_y = [-2, -1.5, -0.5, 0.5, 1.5, 2]
    # Use scipy to smooth if needed, or just plot dashed line
    # ax.plot(traj_x, traj_y, color='black', lw=1.5, linestyle=':', label='Swing Trajectory')

    # 5. Formatting
    ax.set_aspect('equal')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    
    # Remove ticks (Coordinate values matter less than relative shape)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Labels
    ax.set_xlabel("Tip Horizontal Position (X)", fontweight='bold')
    ax.set_ylabel("Tip Vertical Position (Y)", fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_LEFT, label='Torque -1 (Left)'),
        mpatches.Patch(color=COLOR_NOOP, label='Torque 0 (Coast)'),
        mpatches.Patch(color=COLOR_RIGHT, label='Torque +1 (Right)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    save_path = os.path.join(OUTPUT_DIR, "fig_acrobot_workspace.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Mock model for testing if file missing
    model = load_model(MODEL_PATH)
    if model: 
        plot_acrobot_workspace(model)
    else:
        print("Model not found. Cannot generate plot.")