#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from threading import Thread
from multiprocessing.connection import Listener
from typing import Optional
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Import ultralytics failed: {e}")

print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

def set_server_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print(f"Server Seed set to: {seed}")

class PendulumStateEstimator:
    def __init__(self, model_path: str, device: str = "cuda:0", img_size: int = 640, window_size: int = 5):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.window_size = window_size
        print(f"[Estimator] Loading YOLO from {model_path} ...")
        self.model = YOLO(model_path).to(self.device)
        
        # Initialize buffer for moving average and tracking previous states
        self.pos_buffer = deque(maxlen=self.window_size)
        self.prev_smoothed_pos: Optional[np.ndarray] = None
        
        # [NEW] Track the raw keypoints from the previous frame for continuity checking
        self.prev_kpts: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def clone(self):
        new_obj = PendulumStateEstimator.__new__(PendulumStateEstimator)
        new_obj.device = self.device
        new_obj.img_size = self.img_size
        new_obj.window_size = self.window_size
        new_obj.model = self.model
        
        new_obj.pos_buffer = deque(maxlen=self.window_size)
        new_obj.prev_smoothed_pos = None
        new_obj.prev_kpts = None
        return new_obj

    def begin_episode(self):
        # Clear buffer and history at the start of each episode
        self.pos_buffer.clear()
        self.prev_smoothed_pos = None
        self.prev_kpts = None

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        results = self.model.predict(frame_bgr, verbose=False, device=self.device, conf=0.5)
        if not results or len(results) == 0: return None
        r = results[0]
        if r.keypoints is None or r.keypoints.xy.shape[1] < 2: return None

        kpts = r.keypoints.xy[0].cpu().numpy()
        pivot = kpts[0]
        tip = kpts[1]

        # ==========================================
        # [NEW] Continuous Keypoint Tracking & Swap Logic
        # ==========================================
        if self.prev_kpts is not None:
            prev_pivot, prev_tip = self.prev_kpts
            
            # Calculate total pixel movement if we DO NOT swap
            dist_no_swap = np.linalg.norm(pivot - prev_pivot) + np.linalg.norm(tip - prev_tip)
            # Calculate total pixel movement if we DO swap
            dist_swap = np.linalg.norm(tip - prev_pivot) + np.linalg.norm(pivot - prev_tip)
            
            # If swapping results in a significantly smaller movement trajectory,
            # it means YOLO flipped the pivot and tip. We must swap them back.
            if dist_swap < dist_no_swap:
                pivot, tip = tip, pivot # Pythonic swap!
                
        # Save the current corrected keypoints for the next frame
        self.prev_kpts = (pivot, tip)
        # ==========================================

        pivot_x, pivot_y = pivot
        tip_x, tip_y = tip
        
        scale = self.img_size / 2.0 
        norm_x = (tip_x - pivot_x) / scale
        norm_y = (tip_y - pivot_y) / scale
        
        current_raw_pos = np.array([norm_x, norm_y], dtype=np.float32)
        
        # Append strictly CORRECTED raw position to buffer
        self.pos_buffer.append(current_raw_pos)
        
        # Calculate smoothed position (Moving Average)
        smoothed_pos = np.mean(self.pos_buffer, axis=0, dtype=np.float32)
        
        # Calculate velocity based on smoothed positions
        if self.prev_smoothed_pos is None:
            velocity = np.zeros(2, dtype=np.float32)
        else:
            velocity = smoothed_pos - self.prev_smoothed_pos
            
        self.prev_smoothed_pos = smoothed_pos
        
        return np.concatenate([smoothed_pos, velocity])

class InferenceServer:
    def __init__(self, est: PendulumStateEstimator, host="127.0.0.1", port=6000, authkey=b"pendulum-rpc"):
        self.master_est = est
        self.address = (host, port)
        self.authkey = authkey

    def _handle_client(self, conn):
        session = self.master_est.clone()
        try:
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError):
                    break
                except Exception as e:
                    print(f"[RPC] Recv Error: {e}")
                    break

                if msg[0] == "reset":
                    session.begin_episode()
                    conn.send((True, "ok"))
                elif msg[0] == "infer":
                    res = session.process_frame(msg[1])
                    try:
                        conn.send((True, res) if res is not None else (False, "fail"))
                    except (ConnectionResetError, BrokenPipeError):
                        break
        finally:
            conn.close()

    def serve_forever(self):
        l = Listener(self.address, authkey=self.authkey)
        print(f"[RPC] Pendulum Server listening on {self.address}")
        while True:
            Thread(target=self._handle_client, args=(l.accept(),), daemon=True).start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./best.pt")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--window", type=int, default=5, help="Moving average window size")
    args = parser.parse_args()
    
    set_server_seed(42)
    # Pass the window size to the estimator
    est = PendulumStateEstimator(args.model, window_size=args.window)
    InferenceServer(est, port=args.port).serve_forever()