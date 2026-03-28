#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import gymnasium as gym
import cv2
from tqdm import tqdm

# ==========================================
# 1. Environment Wrapper for Visual Clutter
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        self.clutter_texture = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
            
        frame = frame.astype(np.float32)

        if self.add_clutter:
            bg_mask = np.all(frame > 240, axis=-1)
            self.clutter_texture = np.roll(self.clutter_texture, shift=3, axis=0)
            self.clutter_texture = np.roll(self.clutter_texture, shift=7, axis=1)
            h, w = frame.shape[:2]
            texture_resized = cv2.resize(self.clutter_texture, (w, h))
            frame[bg_mask] = texture_resized[bg_mask].astype(np.float32)

        if self.gaussian_std > 0:
            noise = np.random.normal(0, self.gaussian_std, frame.shape)
            frame += noise

        return np.clip(frame, 0, 255).astype(np.uint8)

# ==========================================
# 2. Dataset Generation and Auto-Annotation
# ==========================================
def generate_dataset(num_images=1000, output_dir="cartpole_noisy_dataset"):
    # Create YOLO dataset directory structure
    img_dir = os.path.join(output_dir, "images", "train")
    lbl_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = VisuallyClutteredWrapper(env, gaussian_std=25.0, add_clutter=True)
    
    # CartPole visual rendering constants
    screen_width = 600.0
    screen_height = 400.0
    world_width = 4.8
    scale = screen_width / world_width
    cart_y_pixel = 300.0  # Cart is drawn at Y=300 (from top)
    cart_width = 50.0
    cart_height = 30.0
    pole_width = 10.0
    pole_len = scale * 1.0  # 125 pixels
    
    obs, _ = env.reset()
    
    print(f"[*] Generating {num_images} noisy images and auto-annotations...")
    for i in tqdm(range(num_images)):
        # 1. Take a random action to explore different states
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
            continue
            
        # 2. Get the noisy image
        frame_rgb = env.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 3. Extract exact ground truth state
        x, _, theta, _ = obs
        
        # 4. Calculate Keypoints based on Gym's exact rendering math
        # Pivot (Keypoint 0)
        cx = screen_width / 2.0 + x * scale
        cy = cart_y_pixel
        
        # Tip (Keypoint 1)
        # Note: Gym renders angle positive clockwise from vertical up
        tx = cx + math.sin(theta) * pole_len
        ty = cy - math.cos(theta) * pole_len
        
        # 5. Calculate Bounding Box bounding both cart and pole
        min_x = min(cx - cart_width/2.0, tx - pole_width/2.0)
        max_x = max(cx + cart_width/2.0, tx + pole_width/2.0)
        min_y = min(cy - cart_height/2.0, ty - pole_width/2.0)
        max_y = max(cy + cart_height/2.0, ty + pole_width/2.0)
        
        box_w = max_x - min_x
        box_h = max_y - min_y
        box_cx = min_x + box_w / 2.0
        box_cy = min_y + box_h / 2.0
        
        # 6. Normalize coordinates to [0, 1] for YOLO
        norm_box_cx = np.clip(box_cx / screen_width, 0, 1)
        norm_box_cy = np.clip(box_cy / screen_height, 0, 1)
        norm_box_w = np.clip(box_w / screen_width, 0, 1)
        norm_box_h = np.clip(box_h / screen_height, 0, 1)
        
        norm_cx = np.clip(cx / screen_width, 0, 1)
        norm_cy = np.clip(cy / screen_height, 0, 1)
        norm_tx = np.clip(tx / screen_width, 0, 1)
        norm_ty = np.clip(ty / screen_height, 0, 1)
        
        # 7. Write YOLO Pose Annotation (class_id cx cy w h kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis)
        # Visibility flag '2' means visible and labeled
        annotation = f"0 {norm_box_cx:.6f} {norm_box_cy:.6f} {norm_box_w:.6f} {norm_box_h:.6f} " \
                     f"{norm_cx:.6f} {norm_cy:.6f} 2 {norm_tx:.6f} {norm_ty:.6f} 2\n"
                     
        img_name = f"frame_{i:05d}.jpg"
        lbl_name = f"frame_{i:05d}.txt"
        
        cv2.imwrite(os.path.join(img_dir, img_name), frame_bgr)
        with open(os.path.join(lbl_dir, lbl_name), "w") as f:
            f.write(annotation)
            
    # 8. Create data.yaml for YOLO training
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/train  # Use train for val as it's a simple overfitting task for robustness

nc: 1
names: ['cartpole']
kpt_shape: [2, 3]  # 2 keypoints, each has (x, y, visible)
"""
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(yaml_content.strip())
        
    print(f"\n[*] Dataset generation complete! Saved to '{output_dir}'.")
    print(f"[*] You can now run YOLO training with 'data.yaml'.")

if __name__ == "__main__":
    generate_dataset(num_images=1000)