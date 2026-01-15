#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import traceback
import numpy as np
import torch
from threading import Thread
from multiprocessing.connection import Listener
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Import ultralytics failed: {e}")

print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# -------------------
# Acrobot 8 direction
# -------------------
class AcrobotStateEstimator:
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "cuda:0",
                 img_size: int = 640,
                 _loaded_model: Optional[YOLO] = None):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
        if _loaded_model:
            self.model = _loaded_model
        elif model_path:
            print(f"[Estimator] Loading YOLO from {model_path} ...")
            self.model = YOLO(model_path).to(self.device)
        else:
            raise ValueError("Either model_path or _loaded_model must be provided.")

        self._init_args = locals()
        self.prev_kpts: Optional[np.ndarray] = None

    def clone(self) -> 'AcrobotStateEstimator':
        args = self._init_args.copy()
        args.pop('self')
        args['_loaded_model'] = self.model
        args['model_path'] = None
        return AcrobotStateEstimator(**args)

    def begin_episode(self):
        self.prev_kpts = None

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        # 1. YOLO 推理
        results = self.model.predict(frame_bgr, verbose=False, device=self.device, conf=0.5)
        
        if not results or len(results) == 0:
            return None

        r = results[0]
        if r.keypoints is None or r.keypoints.xy.shape[1] < 3:
            return None
        kpts = r.keypoints.xy[0].cpu().numpy() # shape (3, 2)
        
        base_x, base_y = kpts[0]
        joint_x, joint_y = kpts[1]
        tip_x, tip_y = kpts[2]
        scale = self.img_size / 2.0 
        
        norm_joint_x = (joint_x - base_x) / scale
        norm_joint_y = (joint_y - base_y) / scale
        norm_tip_x = (tip_x - base_x) / scale
        norm_tip_y = (tip_y - base_y) / scale
        current_pos_vec = np.array([norm_joint_x, norm_joint_y, norm_tip_x, norm_tip_y], dtype=np.float32)

        if self.prev_kpts is None:
            velocity_vec = np.zeros(4, dtype=np.float32)
        else:
            velocity_vec = current_pos_vec - self.prev_kpts
            
        self.prev_kpts = current_pos_vec
        
        final_state = np.concatenate([current_pos_vec, velocity_vec])
        
        return final_state

# -------------------
# RPC Server 
# -------------------
class InferenceServer:
    def __init__(self, est: AcrobotStateEstimator, host="127.0.0.1", port=6000, authkey=b"acrobot-rpc"):
        self.master_estimator = est
        self.address = (host, port)
        self.authkey = authkey

    def _handle_client(self, conn):
        print(f"[RPC-SRV] Client connected: {conn}")
        session_est = self.master_estimator.clone()
        
        try:
            while True:
                try:
                    msg = conn.recv()
                except:
                    break
                
                if not isinstance(msg, tuple) or len(msg) < 1:
                    conn.send((False, "bad format")); continue

                cmd = msg[0]
                payload = msg[1] if len(msg) > 1 else None
                
                if cmd == "reset":
                    session_est.begin_episode()
                    conn.send((True, "reset_ok"))
                elif cmd == "infer":
                    if payload is None:
                        conn.send((False, "no image"))
                    else:
                        res = session_est.process_frame(payload)
                        if res is not None:
                            conn.send((True, res))
                        else:
                            conn.send((False, "det failed"))
                else:
                    conn.send((False, "unknown cmd"))
        finally:
            conn.close()

    def serve_forever(self):
        try:
            listener = Listener(self.address, authkey=self.authkey)
        except Exception as e:
            print(f"[FATAL] Bind failed: {e}"); return
        print(f"[RPC-SRV] Acrobot Vision Server running on {self.address}")
        
        while True:
            try:
                conn = listener.accept()
                t = Thread(target=self._handle_client, args=(conn,), daemon=True)
                t.start()
            except Exception as e:
                print(f"Accept error: {e}"); break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="acrobot_runs/train_v1/weights/best.pt")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    master = AcrobotStateEstimator(model_path=args.model, device=args.device)
    server = InferenceServer(master, port=args.port, authkey=b"acrobot-rpc")
    server.serve_forever()