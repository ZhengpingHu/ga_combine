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

# 限制线程冲突，保证计算资源专注
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Import ultralytics failed: {e}")

# 让 print 立即输出
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# -------------------
# 核心逻辑：CartPole 状态估计器
# -------------------
class CartPoleStateEstimator:
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "cuda:0",
                 angle_gain: float = 1.12,  # 我们的核心校准参数
                 smoothing_alpha: float = 1.0, # 1.0 = 无滤波(响应最快)
                 _loaded_model: Optional[YOLO] = None):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. 模型加载逻辑 (支持克隆)
        if _loaded_model:
            self.model = _loaded_model
        elif model_path:
            print(f"[Estimator] Loading model from {model_path} ...")
            self.model = YOLO(model_path).to(self.device)
        else:
            raise ValueError("Either model_path or _loaded_model must be provided.")

        # 保存初始化参数以便 clone
        self._init_args = locals()
        
        # 2. 物理与算法参数
        self.gain = angle_gain
        self.alpha = smoothing_alpha
        self.screen_width = 600
        self.world_width = 4.8
        self.scale = self.screen_width / self.world_width
        
        # 3. 内部状态
        # state 结构: [x, v, theta, omega]
        self.state: Optional[np.ndarray] = None
        self.last_raw_state: Optional[Tuple[float, float]] = None # (x, theta)

    def clone(self) -> 'CartPoleStateEstimator':
        """
        克隆当前估计器逻辑，但共享显存中的 YOLO 模型对象。
        """
        args = self._init_args.copy()
        args.pop('self')
        args['_loaded_model'] = self.model # 传递已加载的模型对象
        args['model_path'] = None
        return CartPoleStateEstimator(**args)

    def begin_episode(self):
        """重置状态，准备新的一局"""
        self.state = None
        self.last_raw_state = None

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        核心推理函数
        Input: 图像数据
        Output: [x, v, theta, omega]
        """
        # 1. YOLO 推理
        # verbose=False 防止刷屏
        results = self.model.predict(frame_bgr, verbose=False, device=self.device)
        
        if not results or len(results) == 0:
            return None # 没检测到东西

        r = results[0]
        if r.keypoints is None or r.keypoints.has_visible is False:
            return None

        # 取置信度最高的一个
        # keypoints.xy 形状是 [N, 2, 2] -> 我们取 [0]
        kpts = r.keypoints.xy[0].cpu().numpy()
        
        if len(kpts) < 2:
            return None

        # 2. 提取坐标
        cx, cy = kpts[0] # 车
        tx, ty = kpts[1] # 杆顶

        # 3. 物理计算 (含 Gain 校准)
        # 位置 x
        raw_x = (cx - (self.screen_width / 2.0)) / self.scale
        
        # 角度 theta
        dx = tx - cx
        dy = ty - cy
        base_angle = math.atan2(dx, -dy)
        raw_theta = base_angle * self.gain # 应用核心校准

        # 4. 状态更新 (滤波 + 差分)
        current_x = raw_x
        current_theta = raw_theta
        
        # 如果 alpha < 1.0，应用 EMA 滤波 (目前建议 1.0)
        if self.last_raw_state is not None:
             last_x_smooth, last_theta_smooth = self.last_raw_state
             current_x = self.alpha * raw_x + (1 - self.alpha) * last_x_smooth
             current_theta = self.alpha * raw_theta + (1 - self.alpha) * last_theta_smooth

        # 计算速度 (Gym dt = 0.02)
        dt = 0.02
        if self.state is None:
            vx = 0.0
            omega = 0.0
        else:
            # state[0] 是上一帧的 x, state[2] 是上一帧的 theta
            vx = (current_x - self.state[0]) / dt
            omega = (current_theta - self.state[2]) / dt

        # 更新内部状态
        self.last_raw_state = (current_x, current_theta)
        self.state = np.array([current_x, vx, current_theta, omega], dtype=np.float32)
        
        return self.state.copy()


# -------------------
# RPC Server (保持了您提供的优秀结构)
# -------------------
class InferenceServer:
    def __init__(self, est: CartPoleStateEstimator, host="127.0.0.1", port=6001, authkey=b"cartpole-rpc"):
        self.master_estimator = est
        self.address = (host, port)
        self.authkey = authkey

    def _handle_client(self, conn):
        print(f"[RPC-SRV] New client connected: {conn}")
        # 为该客户端克隆独立的估计器
        session_est = self.master_estimator.clone()
        
        try:
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError):
                    break # 客户端断开
                except Exception as e:
                    print(f"[RPC-SRV] Recv error: {e}")
                    break

                # 协议检查
                if not isinstance(msg, tuple) or len(msg) < 1:
                    conn.send((False, "bad request format"))
                    continue

                cmd = msg[0]
                payload = msg[1] if len(msg) > 1 else None
                
                try:
                    # --- 命令处理 ---
                    if cmd == "ping":
                        conn.send((True, "pong"))
                    
                    elif cmd == "reset":
                        session_est.begin_episode()
                        conn.send((True, "reset_ok"))
                    
                    elif cmd == "infer":
                        # payload 应该是图像数据 (numpy array)
                        if payload is None:
                            conn.send((False, "missing image data"))
                        else:
                            # 核心：调用估计器
                            result = session_est.process_frame(payload)
                            # 如果 result 是 None (未识别)，发送 False
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
                # 启动线程处理客户端，实现并发
                t = Thread(target=self._handle_client, args=(conn,), daemon=True)
                t.start()
            except Exception as e:
                print(f"[RPC-SRV] Accept failed: {e}")
                break

# -------------------
# 主程序入口
# -------------------
def main():
    parser = argparse.ArgumentParser(description="CartPole YOLO RPC Server")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to best.pt")
    parser.add_argument("--port", type=int, default=6000, help="RPC Port")
    parser.add_argument("--gain", type=float, default=1.12, help="Angle calibration gain")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing alpha (1.0 = no filter)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    args = parser.parse_args()

    # 1. 初始化母版估计器 (加载模型到 GPU)
    master_est = CartPoleStateEstimator(
        model_path=args.model,
        device=args.device,
        angle_gain=args.gain,
        smoothing_alpha=args.alpha
    )
    
    # 2. 启动服务器
    # 注意：authkey 必须是 bytes 类型
    server = InferenceServer(master_est, host="127.0.0.1", port=args.port, authkey=b"cartpole-rpc")
    server.serve_forever()

if __name__ == "__main__":
    main()