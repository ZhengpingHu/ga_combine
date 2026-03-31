#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import gymnasium as gym
from tqdm import tqdm

# ==========================================
# 1. 噪声环境 Wrapper (适配 MountainCar 600x400)
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        # MountainCar 原生分辨率是 600(宽) x 400(高)
        self.clutter_texture = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)

    def render(self):
        frame = self.env.render()
        if frame is None: return None
        frame = frame.astype(np.float32)

        if self.add_clutter:
            # 识别纯色背景 (通常是偏白色的天空)
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
# 2. 关键点与标签计算逻辑 (请与 Clean 数据集严格对齐)
# ==========================================
def get_mountaincar_keypoints(env):
    """
    修改为生成 2 个关键点 (模拟小车的左锚点和右锚点)
    """
    pos = env.unwrapped.state[0]
    
    min_pos = -1.2
    max_pos = 0.6
    world_width = max_pos - min_pos 
    scale = 600 / world_width       
    
    # 小车绝对物理中心
    cart_x = (pos - min_pos) * scale
    terrain_y = np.sin(3 * pos) * 0.45 * scale + 0.55 * scale
    cart_y = 400 - terrain_y 
    
    # MountainCar 在画面中的大致像素宽度大约是 40 像素左右
    # 我们生成左侧(-20)和右侧(+20)两个特征锚点，刚好和你的 HSR 逻辑完美契合
    p0_x = cart_x - 20.0
    p1_x = cart_x + 20.0
    
    return np.array([
        [p0_x, cart_y],  # 第 1 个点：左侧锚点
        [p1_x, cart_y]   # 第 2 个点：右侧锚点
    ])

def convert_to_yolo_pose(kpts, img_w=600, img_h=400, padding=25):
    """将关键点转换为 YOLOv11 Pose 格式"""
    # 以小车为中心生成一个 Bounding Box (宽50，高50左右)
    min_x = np.clip(np.min(kpts[:, 0]) - padding, 0, img_w)
    max_x = np.clip(np.max(kpts[:, 0]) + padding, 0, img_w)
    min_y = np.clip(np.min(kpts[:, 1]) - padding, 0, img_h)
    max_y = np.clip(np.max(kpts[:, 1]) + padding, 0, img_h)
    
    box_w = max_x - min_x
    box_h = max_y - min_y
    center_x = min_x + box_w / 2.0
    center_y = min_y + box_h / 2.0
    
    # 归一化
    n_cx = center_x / img_w
    n_cy = center_y / img_h
    n_w = box_w / img_w
    n_h = box_h / img_h
    
    # 归一化关键点 (2: visible)
    kpts_str = []
    for pt in kpts:
        nx = pt[0] / img_w
        ny = pt[1] / img_h
        kpts_str.append(f"{nx:.6f} {ny:.6f} 2")
        
    kpt_string = " ".join(kpts_str)
    return f"0 {n_cx:.6f} {n_cy:.6f} {n_w:.6f} {n_h:.6f} {kpt_string}"

# ==========================================
# 3. 数据集生成主程序
# ==========================================
def main():
    TOTAL_IMAGES = 1000
    TRAIN_RATIO = 0.8
    
    base_dir = "datasets/mountaincar_noisy"
    os.makedirs(f"{base_dir}/images/train", exist_ok=True)
    os.makedirs(f"{base_dir}/images/val", exist_ok=True)
    os.makedirs(f"{base_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{base_dir}/labels/val", exist_ok=True)
    
    base_env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = VisuallyClutteredWrapper(base_env, gaussian_std=25.0, add_clutter=True)
    
    env.reset()
    
    print(f"[*] 开始生成 {TOTAL_IMAGES} 张 MountainCar 噪声图像...")
    for i in tqdm(range(TOTAL_IMAGES)):
        action = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(action)
        
        if done or truncated:
            env.reset()
            continue
            
        frame = env.render()
        kpts = get_mountaincar_keypoints(env)
        yolo_label = convert_to_yolo_pose(kpts)
        
        split = "train" if i < (TOTAL_IMAGES * TRAIN_RATIO) else "val"
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        img_path = f"{base_dir}/images/{split}/frame_{i:05d}.jpg"
        label_path = f"{base_dir}/labels/{split}/frame_{i:05d}.txt"
        
        cv2.imwrite(img_path, frame_bgr)
        with open(label_path, "w") as f:
            f.write(yolo_label)

    env.close()
    print("[*] 噪声数据集生成完毕！")

if __name__ == "__main__":
    main()