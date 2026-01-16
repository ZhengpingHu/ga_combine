# -*- coding: utf-8 -*-
# save as: plot_acrobot_policy.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= Config =================
MODEL_PATH = "./best_model_101.npz"
OUTPUT_DIR = "policy_visualization_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grid Resolution
RESOLUTION = 100 # 100x100 grid

# IEEE Styling
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10
})

# ================= Model & Physics Helper =================
class FixedNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Acrobot: 8 inputs -> 3 outputs (Action: -1, 0, 1)
        self.net = nn.Linear(8, 3, bias=True)
    def forward(self, x): return self.net(x)

def load_model(path):
    if not os.path.exists(path): return None
    data = np.load(path)
    model = FixedNNPolicy()
    # Loading weights logic (simplified)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(data['weights'][offset:offset+n]).view_as(p))
            offset += n
    model.eval()
    return model

def get_8d_input_from_angles(theta1, theta2):
    """
    Simulate the Visual Input (8D) from physical angles.
    Assumption: Input is [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    Links length l1=1.0, l2=1.0 (Approx standard)
    """
    # Kinematics
    l1, l2 = 1.0, 1.0
    
    # Joint 1 (Elbow) position
    x1 = -l1 * np.sin(theta1)
    y1 = l1 * np.cos(theta1)
    
    # Joint 2 (Tip) position
    x2 = x1 - l2 * np.sin(theta1 + theta2)
    y2 = y1 + l2 * np.cos(theta1 + theta2)
    
    # Velocities (Assume static for decision map)
    vx1, vy1, vx2, vy2 = 0.0, 0.0, 0.0, 0.0
    
    # Normalize if your training did (Standard Acrobot obs are roughly -1 to 1)
    # Here we feed raw normalized coords. 
    # You might need to adjust scale factor if your HSR used one (e.g. / width).
    return torch.tensor([[x1, y1, x2, y2, vx1, vy1, vx2, vy2]], dtype=torch.float32)

# ================= Plotting =================
def plot_acrobot_policy(model):
    # 1. Create Grid
    t1 = np.linspace(-np.pi, np.pi, RESOLUTION)
    t2 = np.linspace(-np.pi, np.pi, RESOLUTION)
    T1, T2 = np.meshgrid(t1, t2)
    
    Actions = np.zeros_like(T1)

    print("Scanning decision landscape...")
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            theta1 = T1[i, j]
            theta2 = T2[i, j]
            
            # Construct Input
            state = get_8d_input_from_angles(theta1, theta2)
            
            # Inference
            with torch.no_grad():
                logits = model(state)
                action = torch.argmax(logits).item() # 0, 1, 2
            
            Actions[i, j] = action

    # 2. Plot Heatmap
    fig, ax = plt.subplots(figsize=(4.0, 3.5), dpi=300)
    
    # Colormap: 0=Red(-1), 1=Gray(0), 2=Blue(+1)
    # Note: Check your action mapping! Usually in Gym: 0=-1, 1=0, 2=+1
    cmap = sns.color_palette(["#D62728", "#E0E0E0", "#1F77B4"]) 
    
    c = ax.pcolormesh(T1, T2, Actions, cmap=plt.cm.colors.ListedColormap(cmap), shading='auto')
    
    # Decoration
    ax.set_xlabel(r"Shoulder Angle ($\theta_1$)", fontweight='bold')
    ax.set_ylabel(r"Elbow Angle ($\theta_2$)", fontweight='bold')
    
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    
    # Axis Ticks (Pi symbols)
    ticks = [-np.pi, 0, np.pi]
    labels = [r'$-\pi$', r'$0$', r'$\pi$']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#D62728', lw=4, label='Torque -1 (Left)'),
        Line2D([0], [0], color='#E0E0E0', lw=4, label='Torque 0 (No-Op)'),
        Line2D([0], [0], color='#1F77B4', lw=4, label='Torque +1 (Right)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_acrobot_policy.pdf"), format='pdf')
    print("Done.")

if __name__ == "__main__":
    # Ensure you have the model file
    model = load_model(MODEL_PATH)
    if model: plot_acrobot_policy(model)