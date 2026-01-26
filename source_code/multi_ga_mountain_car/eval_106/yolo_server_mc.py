#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# YOLO Server for MountainCar (2-Point Texture Anchor)

import os
import argparse
import numpy as np
import torch
from threading import Thread
from multiprocessing.connection import Listener
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError("Please install ultralytics: pip install ultralytics")

print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

class MountainCarStateEstimator:
    def __init__(self, model_path, device="cuda:0", _loaded_model=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if _loaded_model:
            self.model = _loaded_model
        else:
            print(f" [Vision] Loading YOLO model: {model_path} ...")
            self.model = YOLO(model_path).to(self.device)
            
        self.min_pos = -1.2
        self.max_pos = 0.6
        self.screen_width = 600
        self.scale = self.screen_width / (self.max_pos - self.min_pos)
        
        self.last_pos = None
        self.state = None

        self._init_args = locals() 

    def clone(self):
        args = self._init_args.copy()
        args.pop('self')
        args['_loaded_model'] = self.model
        return MountainCarStateEstimator(**args)

    def begin_episode(self):
        self.last_pos = None
        self.state = np.zeros(2, dtype=np.float32) # [pos, vel]

    @torch.no_grad()
    def process_frame(self, frame_bgr):
        results = self.model.predict(frame_bgr, verbose=False, device=self.device)
        if not results or not results[0].keypoints:
            return self.state 
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        if kpts.shape[0] < 2:
            return self.state
        center_x = (kpts[0][0] + kpts[1][0]) / 2.0

        current_pos = (center_x / self.scale) + self.min_pos
        current_pos = np.clip(current_pos, self.min_pos, self.max_pos)

        if self.last_pos is None:
            vel = 0.0
        else:
            vel = current_pos - self.last_pos
            
        self.last_pos = current_pos
        self.state = np.array([current_pos, vel], dtype=np.float32)
        
        return self.state.copy()

class InferenceServer:
    def __init__(self, est, host="127.0.0.1", port=6001, authkey=b"mc-rpc"):
        self.master_est = est
        self.addr = (host, port)
        self.authkey = authkey

    def _handle(self, conn):
        sess = self.master_est.clone()
        try:
            while True:
                msg = conn.recv()
                if not msg: break
                cmd, payload = msg[0], msg[1] if len(msg)>1 else None
                
                if cmd == "reset":
                    sess.begin_episode()
                    conn.send((True, "ok"))
                elif cmd == "infer":
                    res = sess.process_frame(payload)
                    conn.send((True, res) if res is not None else (False, None))
                else:
                    conn.send((False, "unknown"))
        except: pass
        finally: conn.close()

    def serve(self):
        listener = Listener(self.addr, authkey=self.authkey)
        print(f"[RPC] MountainCar Vision Server running on {self.addr[0]}:{self.addr[1]}")
        print(f"Waiting for GA clients...")
        while True:
            Thread(target=self._handle, args=(listener.accept(),), daemon=True).start()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--port", type=int, default=6001)
    args = parser.parse_args()
    
    est = MountainCarStateEstimator(args.model)
    InferenceServer(est, port=args.port).serve()

if __name__ == "__main__":
    main()