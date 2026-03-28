#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
import gymnasium as gym
import torch
from ultralytics import YOLO

# Set environment variables for performance
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ==========================================
# 1. Environment Wrapper for Visual Clutter
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        """
        Injects dynamic background clutter and Gaussian noise to test YOLO robustness.
        """
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        
        # Pre-generate a static random noise texture for the background
        self.clutter_texture = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)

    def render(self):
        # Capture the original clean RGB frame
        frame = self.env.render()
        if frame is None:
            return None
            
        frame = frame.astype(np.float32)

        # Inject Dynamic Background Clutter
        if self.add_clutter:
            # CartPole's background is nearly pure white. Mask pixels > 240.
            bg_mask = np.all(frame > 240, axis=-1)
            
            # Roll texture to create a dynamic/moving background effect
            self.clutter_texture = np.roll(self.clutter_texture, shift=5, axis=0)
            self.clutter_texture = np.roll(self.clutter_texture, shift=5, axis=1)
            
            h, w = frame.shape[:2]
            texture_resized = cv2.resize(self.clutter_texture, (w, h))
            
            # Replace pure white background with clutter
            frame[bg_mask] = texture_resized[bg_mask].astype(np.float32)

        # Inject Gaussian Sensor Noise
        if self.gaussian_std > 0:
            noise = np.random.normal(0, self.gaussian_std, frame.shape)
            frame += noise

        # Clip and convert back to uint8
        return np.clip(frame, 0, 255).astype(np.uint8)

# ==========================================
# 2. Main Visual Testing Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="YOLO Visual Robustness Tester")
    parser.add_argument("--model", type=str, default="best_noise.pt", help="Path to your YOLO best.pt")
    parser.add_argument("--noise-std", type=float, default=25.0, help="Gaussian noise standard deviation")
    parser.add_argument("--no-clutter", action="store_true", help="Disable dynamic background clutter")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading YOLO model from {args.model} onto {device}...")
    try:
        model = YOLO(args.model).to(device)
    except Exception as e:
        print(f"[FATAL] Failed to load model: {e}")
        return

    print(f"[*] Initializing noisy CartPole environment...")
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # Wrap the environment to inject noise
    env = VisuallyClutteredWrapper(
        base_env, 
        gaussian_std=args.noise_std, 
        add_clutter=not args.no_clutter
    )

    print("[*] Starting visual test. Press 'q' to quit.")
    
    obs, _ = env.reset()
    
    try:
        while True:
            # 1. Render the NOISY frame (RGB format)
            frame_rgb = env.render()
            if frame_rgb is None:
                break
                
            # Convert RGB to BGR for YOLO inference and OpenCV display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 2. YOLO Inference
            # conf=0.5 ensures we only draw confident detections
            results = model.predict(frame_bgr, device=device, verbose=False, conf=0.5)
            
            # 3. Draw keypoints and bounding boxes using YOLO's built-in plotter
            if results and len(results) > 0:
                # plot() returns a BGR numpy array with all annotations drawn
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame_bgr # Fallback if nothing detected
                
            # 4. Display the frame on screen
            cv2.imshow("YOLO Robustness Test (Press 'q' to exit)", annotated_frame)
            
            # Wait 20ms (simulate 50 FPS). Break loop if 'q' is pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            # 5. Take a random action to keep the pole moving
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reset if the pole falls
            if terminated or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n[*] Visual test interrupted by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()
        print("[*] Test finished.")

if __name__ == "__main__":
    main()