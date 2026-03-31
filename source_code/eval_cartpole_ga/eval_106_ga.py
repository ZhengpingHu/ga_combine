#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO

# ==========================================
# [新增] 引入与训练时完全一致的 PyTorch 网络结构
# ==========================================
class FixedNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ==========================================
# [配置区] 路径与参数
# ==========================================
GA_WEIGHT_BASE_PATH = "./ga_seed" 
SEEDS = [101, 102, 103, 104, 105]

# 识别模型路径
YOLO_MODELS = {
    "Zero-shot": "./best.pt",       # Clean 训练的模型
    "Finetuned": "./best_noise.pt"  # 噪声微调后的模型
}

TEST_BASE_SEED = 106
TEST_ENV_COUNT = 100

# ==========================================
# 1. 环境 Wrapper (保持与 PPO 评估严格一致)
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        # 随机背景纹理 (CartPole 600x400)
        self.clutter_texture = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)

    def render(self):
        frame = self.env.render()
        if frame is None: return None
        frame = frame.astype(np.float32)

        if self.add_clutter:
            bg_mask = np.all(frame > 240, axis=-1)
            # 模拟背景滚动增加动态干扰
            self.clutter_texture = np.roll(self.clutter_texture, shift=3, axis=0)
            h, w = frame.shape[:2]
            texture_resized = cv2.resize(self.clutter_texture, (w, h))
            frame[bg_mask] = texture_resized[bg_mask].astype(np.float32)

        if self.gaussian_std > 0:
            noise = np.random.normal(0, self.gaussian_std, frame.shape)
            frame += noise

        return np.clip(frame, 0, 255).astype(np.uint8)

# ==========================================
# 2. HSR 状态提取逻辑
# ==========================================
def get_hsr_state(model, frame, device):
    """
    极简 HSR: 仅从小车与杆尖位置提取 X 坐标
    """
    results = model.predict(frame, verbose=False, device=device, conf=0.2)
    if not results or len(results[0].keypoints) == 0:
        return np.zeros(4)
    
    kpts = results[0].keypoints.xy[0].cpu().numpy()
    if kpts.shape[0] < 2: return np.zeros(4)
    
    # 归一化小车 X 位置 (假设画面中心 300 为 0)
    cart_x = (kpts[0][0] - 300) / 300.0
    # 简化状态，仅用于验证闭环可行性
    return np.array([cart_x, 0.0, 0.0, 0.0], dtype=np.float32)

# ==========================================
# 3. 核心评估函数
# ==========================================
def run_evaluation(model_tag, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] 正在加载 YOLO [{model_tag}] 模型: {model_path}")
    yolo_model = YOLO(model_path).to(device)
    
    # 统一生成 100 个未见过的测试种子
    rng = np.random.default_rng(TEST_BASE_SEED)
    test_seeds = rng.integers(0, 2**31-1, TEST_ENV_COUNT)
    
    all_data = []

    for seed in SEEDS:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tag = f"ga_cartpole_{timestamp}_seed{seed}"
        
        weight_file = f"{GA_WEIGHT_BASE_PATH}{seed}/best_model_full_eval.npz"
        if not os.path.exists(weight_file):
            print(f"[!] 找不到种子 {seed} 的权重文件: {weight_file}")
            continue
        
        # --------------------------------------------------
        # [修改点] 健壮解析 .npz 并加载到 PyTorch Policy 中
        # --------------------------------------------------
        try:
            npz_data = np.load(weight_file)
            # 无论保存的是独立数组还是字典形式，全部提取并压平
            flat_params = np.concatenate([npz_data[k].flatten() for k in npz_data.files])
            
            # 实例化 PyTorch 控制器
            policy = FixedNNPolicy()
            
            if len(flat_params) == 10:
                # 注意：PyTorch 的 nn.Linear(4,2) weight 形状是 [2, 4]，这里必须确保形状匹配
                policy.net.weight.data = torch.FloatTensor(flat_params[:8].reshape(2, 4))
                policy.net.bias.data = torch.FloatTensor(flat_params[8:])
            else:
                print(f"[!] 警告: 参数长度 {len(flat_params)} 不等于 10，Agent {seed} 跳过！")
                continue
                
        except Exception as e:
            print(f"[!] 读取或加载模型时出错 ({weight_file}): {e}")
            continue
            
        policy.eval() # 切换至评估模式
        policy.to("cpu") # 控制器太小了，在 CPU 跑推理甚至比转移到 GPU 更快
        
        print(f"--- 评估 Agent {seed} ({model_tag}) ---")
        for env_seed in tqdm(test_seeds, desc=f"Seed {seed}"):
            env = gym.make("CartPole-v1", render_mode="rgb_array")
            env = VisuallyClutteredWrapper(env, gaussian_std=25.0, add_clutter=True)
            _ = env.reset(seed=int(env_seed))
            
            total_reward = 0
            done = False
            while not done:
                frame = env.render()
                # 1. 前端 YOLO + HSR 感知
                state_np = get_hsr_state(yolo_model, frame, device)
                
                # 2. 转换为 PyTorch Tensor (添加 Batch 维度 [1, 4])
                state_tensor = torch.FloatTensor(state_np).unsqueeze(0)
                
                # 3. 后端 Policy 动作推理 (完全模拟训练环境)
                with torch.no_grad():
                    action_logits = policy(state_tensor)
                    action = action_logits.argmax(dim=1).item()
                
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            all_data.append({
                "Agent_Source_Seed": str(seed),
                "Reward": total_reward,
                "Model_Type": model_tag
            })
            env.close()
            
    return pd.DataFrame(all_data)

# ==========================================
# 4. 绘图与主程序
# ==========================================
def plot_results(df, model_tag):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "axes.labelsize": 10, "font.size": 10, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linestyle": "--", "figure.constrained_layout.use": True
    })
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    
    # Violin
    sns.violinplot(data=df, y="Reward", color="#ECEFF1", inner=None, linewidth=0, alpha=0.6, bw_adjust=0.2, ax=ax, cut=0)
    # Box
    sns.boxplot(data=df, y="Reward", width=0.15, boxprops={'facecolor': 'none', 'edgecolor': '#333333'}, showfliers=False, ax=ax)
    # Strip
    sns.stripplot(data=df, x="Agent_Source_Seed", y="Reward", hue="Agent_Source_Seed", palette="deep", alpha=0.6, jitter=0.3, ax=ax)

    ax.set_title(f"GA CartPole: {model_tag} YOLO in Noisy Env", fontweight='bold')
    ax.set_ylim(0, 520)
    ax.axhline(500, color='#D32F2F', linestyle='-', linewidth=1.0, alpha=0.4)
    
    out_name = f"ga_cartpole_eval_{model_tag.lower()}.pdf"
    plt.savefig(out_name, bbox_inches='tight')
    print(f"[+] 图像已保存: {out_name}")

if __name__ == "__main__":
    os.makedirs("evaluation_ga_results", exist_ok=True)
    
    # 运行两组实验
    for tag, path in YOLO_MODELS.items():
        if not os.path.exists(path):
            print(f"[!] 找不到 YOLO 权重文件 {path}，跳过 {tag} 评估。")
            continue
            
        results_df = run_evaluation(tag, path)
        if not results_df.empty:
            results_df.to_csv(f"evaluation_ga_results/ga_cartpole_{tag.lower()}.csv", index=False)
            plot_results(results_df, tag)