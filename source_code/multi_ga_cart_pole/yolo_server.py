#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
import traceback
import numpy as np
import torch
from threading import Thread
from multiprocessing.connection import Listener
from typing import Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Import ultralytics failed: {e}")

print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

class CartPoleStateEstimator:
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "cuda:0",
                 angle_gain: float = 1.12,  
                 smoothing_alpha: float = 1.0,
                 _loaded_model: Optional[YOLO] = None):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if _loaded_model:
            self.model = _loaded_model
        elif model_path:
            print(f"[Estimator] Loading model from {model_path} ...")
            self.model = YOLO(model_path).to(self.device)
        else:
            raise ValueError("Either model_path or _loaded_model must be provided.")

        self._init_args = locals()
        
        self.gain = angle_gain
        self.alpha = smoothing_alpha
        self.screen_width = 600
        self.world_width = 4.8
        self.scale = self.screen_width / self.world_width
        
        self.state: Optional[np.ndarray] = None
        self.last_raw_state: Optional[Tuple[float, float]] = None # (x, theta)

    def clone(self) -> 'CartPoleStateEstimator':
        args = self._init_args.copy()
        args.pop('self')
        args['_loaded_model'] = self.model
        args['model_path'] = None
        return CartPoleStateEstimator(**args)

    def begin_episode(self):
        self.state = None
        self.last_raw_state = None

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        results = self.model.predict(frame_bgr, verbose=False, device=self.device)
        
        if not results or len(results) == 0:
            return None

        r = results[0]
        if r.keypoints is None or r.keypoints.has_visible is False:
            return None

        kpts = r.keypoints.xy[0].cpu().numpy()
        
        if len(kpts) < 2:
            return None

        cx, cy = kpts[0]
        tx, ty = kpts[1]

        raw_x = (cx - (self.screen_width / 2.0)) / self.scale
        
        dx = tx - cx
        dy = ty - cy
        base_angle = math.atan2(dx, -dy)
        raw_theta = base_angle * self.gain

        current_x = raw_x
        current_theta = raw_theta

        if self.last_raw_state is not None:
             last_x_smooth, last_theta_smooth = self.last_raw_state
             current_x = self.alpha * raw_x + (1 - self.alpha) * last_x_smooth
             current_theta = self.alpha * raw_theta + (1 - self.alpha) * last_theta_smooth

        dt = 0.02
        if self.state is None:
            vx = 0.0
            omega = 0.0
        else:
            vx = (current_x - self.state[0]) / dt
            omega = (current_theta - self.state[2]) / dt

        self.last_raw_state = (current_x, current_theta)
        self.state = np.array([current_x, vx, current_theta, omega], dtype=np.float32)
        
        return self.state.copy()

class InferenceServer:
    def __init__(self, est: CartPoleStateEstimator, host="127.0.0.1", port=6001, authkey=b"cartpole-rpc"):
        self.master_estimator = est
        self.address = (host, port)
        self.authkey = authkey

    def _handle_client(self, conn):
        print(f"[RPC-SRV] New client connected: {conn}")
        session_est = self.master_estimator.clone()
        
        try:
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError):
                    break
                except Exception as e:
                    print(f"[RPC-SRV] Recv error: {e}")
                    break

                if not isinstance(msg, tuple) or len(msg) < 1:
                    conn.send((False, "bad request format"))
                    continue

                cmd = msg[0]
                payload = msg[1] if len(msg) > 1 else None
                
                try:
                    if cmd == "ping":
                        conn.send((True, "pong"))
                    
                    elif cmd == "reset":
                        session_est.begin_episode()
                        conn.send((True, "reset_ok"))
                    
                    elif cmd == "infer":
                        if payload is None:
                            conn.send((False, "missing image data"))
                        else:
                            result = session_est.process_frame(payload)
                            if result is not None:
                                conn.send((True, result))
                            else:
                                conn.send((False, "detection failed"))
                                
                    else:
                        conn.send((False, f"unknown command: {cmd}"))
                        
                except Exception as e:
                    print(f"[RPC-SRV] Processing Error: {e}")
                    traceback.print_exc()
                    conn.send((False, str(e)))
                    break

        finally:
            conn.close()
            print("[RPC-SRV] Client disconnected.")

    def serve_forever(self):
        try:
            listener = Listener(self.address, authkey=self.authkey)
        except OSError as e:
            print(f"[FATAL] Cannot bind to {self.address}. Is port used? Error: {e}")
            return
            
        print(f"[RPC-SRV] Server listening on {self.address} ...")
        print(f"[RPC-SRV] Model: YOLOv11-Pose | Gain: {self.master_estimator.gain}")
        
        while True:
            try:
                conn = listener.accept()
                t = Thread(target=self._handle_client, args=(conn,), daemon=True)
                t.start()
            except Exception as e:
                print(f"[RPC-SRV] Accept failed: {e}")
                break


def main():
    parser = argparse.ArgumentParser(description="CartPole YOLO RPC Server")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to best.pt")
    parser.add_argument("--port", type=int, default=6000, help="RPC Port")
    parser.add_argument("--gain", type=float, default=1.12, help="Angle calibration gain")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing alpha (1.0 = no filter)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    args = parser.parse_args()

    master_est = CartPoleStateEstimator(
        model_path=args.model,
        device=args.device,
        angle_gain=args.gain,
        smoothing_alpha=args.alpha
    )
    

    server = InferenceServer(master_est, host="127.0.0.1", port=args.port, authkey=b"cartpole-rpc")
    server.serve_forever()

if __name__ == "__main__":
    main()