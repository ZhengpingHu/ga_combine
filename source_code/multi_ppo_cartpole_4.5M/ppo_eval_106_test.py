import os
import gymnasium as gym
import cv2
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces

# ==========================================
# 1. 相同的 Pixel Wrapper (确保视觉输入一致)
# ==========================================
class PixelCartPole(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.img_size = (84, 84)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.uint8
        )

    def observation(self, obs):
        img = self.env.render()
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        return img

if __name__ == "__main__":
    train_seeds = [101, 102, 103, 104, 105]
    test_seeds = list(range(106, 206))  # 106 到 205 (共100个独立种子)
    
    all_results = []

    print("=== 开始 PPO 模型的 100-Seed 独立泛化测试 ===")
    
    for train_seed in train_seeds:
        model_path = f"ppo_results_seed_{train_seed}/final_ppo_model.zip"
        if not os.path.exists(model_path):
            print(f"[警告] 找不到模型文件: {model_path}")
            continue
            
        print(f"\n[*] 加载 PPO 模型 (训练种子: {train_seed})...")
        model = PPO.load(model_path, device="cuda")
        
        # 遍历 100 个测试种子
        for test_seed in test_seeds:
            env = gym.make("CartPole-v1", render_mode="rgb_array")
            env = PixelCartPole(env)
            
            obs, _ = env.reset(seed=test_seed)
            done = False
            truncated = False
            total_reward = 0.0
            
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                
            all_results.append({
                "Train_Seed": train_seed,
                "Test_Seed": test_seed,
                "Reward": total_reward
            })
            env.close()
            
        print(f"    完成该模型的 100 次测试！")

    # 保存为 CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("ppo_100_unseen_seeds_evaluation.csv", index=False)
    print("\n=== 测试完成！所有泛化数据已保存至 ppo_100_unseen_seeds_evaluation.csv ===")