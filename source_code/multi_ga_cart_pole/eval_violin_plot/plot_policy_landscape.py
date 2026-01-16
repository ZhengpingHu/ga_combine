# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ==========================================
# 1. Global Configuration & IEEE Styling
# ==========================================
BEST_MODEL_PATH = "./best_model_101.npz"
OUTPUT_DIR = "policy_visualization_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sampling Configuration
X_STEPS = 20        # Number of positions on the track
X_RANGE = (-2.0, 2.0)

THETA_STEPS = 35    # Density of angles per position
THETA_RANGE = (-0.21, 0.21) # Approx +/- 12 degrees

POLE_LENGTH = 0.5   # Visual length of the pole arrows

# IEEE Publication Style Settings
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
    "axes.grid": False,          # Disable grid for this specific plot (cleaner look)
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,   # Remove left spine for physical viz
    "figure.constrained_layout.use": True
})

# IEEE Recommended Colors
COLOR_ACTION_LEFT = '#d62728'  # Brick Red (Push Left)
COLOR_ACTION_RIGHT = '#1f77b4' # Royal Blue (Push Right)

# ==========================================
# 2. Model Definition
# ==========================================
class FixedNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def set_weights_vector(m, vec):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

def load_single_model(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    data = np.load(path)
    model = FixedNNPolicy()
    set_weights_vector(model, data['weights'])
    model.eval()
    return model

# ==========================================
# 3. Plotting Logic
# ==========================================
def plot_physical_arrows(model):
    # Create figure (Single column width or wider for detail)
    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=300)

    # Generate sampling grid
    x_locs = np.linspace(X_RANGE[0], X_RANGE[1], X_STEPS)
    thetas = np.linspace(THETA_RANGE[0], THETA_RANGE[1], THETA_STEPS)
    
    print(f"Generating physical overlay: {len(x_locs)} positions x {len(thetas)} angles...")

    # Iterate through positions and angles (Fan Scan)
    for x in x_locs:
        for theta in thetas:
            # Construct input state: [x, x_dot=0, theta, theta_dot=0]
            state = torch.tensor([[x, 0.0, theta, 0.0]], dtype=torch.float32)
            
            # Model inference
            with torch.no_grad():
                logits = model(state)
                action = torch.argmax(logits).item()
            
            # Determine Color based on Action
            # Action 0 = Left (Red), Action 1 = Right (Blue)
            color = COLOR_ACTION_RIGHT if action == 1 else COLOR_ACTION_LEFT
            
            # Calculate Arrow Coordinates
            # Base of arrow: (x, 0)
            # Tip of arrow: (x + L*sin(theta), L*cos(theta))
            start_x, start_y = x, 0
            
            # CartPole definition: 0 deg is vertical, +theta is right tilt
            end_x = x + POLE_LENGTH * np.sin(theta)
            end_y = POLE_LENGTH * np.cos(theta)
            
            # Draw Arrow
            # Using alpha=0.4 to handle density overlapping
            ax.annotate('', 
                        xy=(end_x, end_y), 
                        xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle='->', 
                                        color=color, 
                                        lw=1.2, 
                                        alpha=0.4))

    # Draw the Track (Ground line)
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=10)
    
    # Aesthetic Adjustments
    ax.set_xlabel("Cart Horizontal Position (m)", fontweight='bold')
    ax.set_ylabel("Pole Angle Representation")
    
    # Remove Y ticks as they represent geometric projection, not absolute units
    ax.set_yticks([])
    
    # Crucial: Set Aspect Ratio to Equal to prevent angle distortion
    ax.set_aspect('equal')
    
    # Set plotting limits
    ax.set_xlim(X_RANGE[0] - 0.2, X_RANGE[1] + 0.2)
    ax.set_ylim(-0.1, POLE_LENGTH * 1.1)
    
    # Custom Legend (Top placement, no frame)
    legend_elements = [
        mpatches.Patch(color=COLOR_ACTION_LEFT, label='Action: Push Left'),
        mpatches.Patch(color=COLOR_ACTION_RIGHT, label='Action: Push Right')
    ]
    
    # Place legend above the plot to avoid obstructing data
    ax.legend(handles=legend_elements, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 1.02),
              ncol=2, 
              frameon=False,
              borderaxespad=0)

    # Save outputs
    save_path_png = os.path.join(OUTPUT_DIR, "fig_policy_arrows.png")
    save_path_pdf = os.path.join(OUTPUT_DIR, "fig_policy_arrows.pdf")
    
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Visualization saved to:")
    print(f"  [PDF] {save_path_pdf}")
    print(f"  [PNG] {save_path_png}")

if __name__ == "__main__":
    model = load_single_model(BEST_MODEL_PATH)
    if model:
        plot_physical_arrows(model)