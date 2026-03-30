#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import gymnasium as gym
from tqdm import tqdm

# ==========================================
# 1. 噪声环境 Wrapper (与评估时完全一致)
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        # Acrobot 原生分辨率是 500x500
        self.clutter_texture = np.random.randint(50, 200, (500, 500, 3), dtype=np.uint8)

    def render(self):
        frame = self.env.render()
        if frame is None: return None
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
# 2. 关键点与标签计算逻辑
# ==========================================
def get_acrobot_keypoints(env):
    """
    ⚠️ 警告：请确保这里的关键点计算逻辑与你生成 Clean 数据集时一模一样！
    下面是基于 Gym 标准 Acrobot-v1 (500x500) 的物理坐标映射。
    """
    state = env.unwrapped.state
    theta1 = state[0]
    theta2 = state[1]
    
    # Acrobot 物理参数与渲染缩放
    L1 = 1.0
    L2 = 1.0
    bound = 2.2
    scale = 500 / (bound * 2) # 约 113.636
    
    # 1. 顶部转轴 (固定在画面中心)
    p0_x, p0_y = 250.0, 250.0 
    
    # 2. 中间肘关节
    # 注意：Gym Acrobot 0度是垂直向下，Y轴在 Pygame 中向下为正
    p1_x = p0_x + scale * L1 * np.sin(theta1)
    p1_y = p0_y + scale * L1 * np.cos(theta1)
    
    # 3. 底部杆尖
    p2_x = p1_x + scale * L2 * np.sin(theta1 + theta2)
    p2_y = p1_y + scale * L2 * np.cos(theta1 + theta2)
    
    return np.array([
        [p0_x, p0_y], 
        [p1_x, p1_y], 
        [p2_x, p2_y]
    ])

def convert_to_yolo_pose(kpts, img_w=500, img_h=500, padding=20):
    """将关键点转换为 YOLOv11 Pose 格式 (类别 x_center y_center w h px py p_vis ...)"""
    # 提取 Bounding Box
    min_x = np.clip(np.min(kpts[:, 0]) - padding, 0, img_w)
    max_x = np.clip(np.max(kpts[:, 0]) + padding, 0, img_w)
    min_y = np.clip(np.min(kpts[:, 1]) - padding, 0, img_h)
    max_y = np.clip(np.max(kpts[:, 1]) + padding, 0, img_h)
    
    box_w = max_x - min_x
    box_h = max_y - min_y
    center_x = min_x + box_w / 2.0
    center_y = min_y + box_h / 2.0
    
    # 归一化 Box 坐标
    n_cx = center_x / img_w
    n_cy = center_y / img_h
    n_w = box_w / img_w
    n_h = box_h / img_h
    
    # 归一化关键点并加上可见度 (2: visible)
    kpts_str = []
    for pt in kpts:
        nx = pt[0] / img_w
        ny = pt[1] / img_h
        kpts_str.append(f"{nx:.6f} {ny:.6f} 2")
        
    kpt_string = " ".join(kpts_str)
    # 类别设为 0
    return f"0 {n_cx:.6f} {n_cy:.6f} {n_w:.6f} {n_h:.6f} {kpt_string}"

# ==========================================
# 3. 数据集生成主程序
# ==========================================
def main():
    TOTAL_IMAGES = 1000
    TRAIN_RATIO = 0.8
    
    base_dir = "datasets/acrobot_noisy"
    os.makedirs(f"{base_dir}/images/train", exist_ok=True)
    os.makedirs(f"{base_dir}/images/val", exist_ok=True)
    os.makedirs(f"{base_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{base_dir}/labels/val", exist_ok=True)
    
    base_env = gym.make("Acrobot-v1", render_mode="rgb_array")
    env = VisuallyClutteredWrapper(base_env, gaussian_std=25.0, add_clutter=True)
    
    env.reset()
    
    print(f"[*] 开始生成 {TOTAL_IMAGES} 张 Acrobot 噪声图像...")
    for i in tqdm(range(TOTAL_IMAGES)):
        # 为了保证姿态多样性，使用随机动作 (或者你可以跑一段 PPO 策略来采集真实分布)
        action = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(action)
        
        if done or truncated:
            env.reset()
            continue
            
        frame = env.render()
        kpts = get_acrobot_keypoints(env)
        yolo_label = convert_to_yolo_pose(kpts)
        
        # 划分训练集与验证集
        split = "train" if i < (TOTAL_IMAGES * TRAIN_RATIO) else "val"
        
        # 从 RGB 转为 BGR 保存
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        img_path = f"{base_dir}/images/{split}/frame_{i:05d}.jpg"
        label_path = f"{base_dir}/labels/{split}/frame_{i:05d}.txt"
        
        cv2.imwrite(img_path, frame_bgr)
        with open(label_path, "w") as f:
            f.write(yolo_label)

    env.close()
    print("[*] 噪声数据集生成完毕！保存在 datasets/acrobot_noisy 目录下。")

if __name__ == "__main__":
    main()