import gymnasium as gym
import numpy as np
import pandas as pd

def analyze_initial_states(env_id, num_samples=10000):
    print(f"\n=== 分析 {env_id} 的初始状态 (Reset) 分布 ===")
    env = gym.make(env_id)
    states = []
    
    for seed in range(num_samples):
        # 强制使用不同的 seed 保证随机性
        obs, _ = env.reset(seed=seed)
        states.append(obs)
        
    states = np.array(states)
    
    # 获取各个维度的最小值、最大值和标准差
    mins = np.min(states, axis=0)
    maxs = np.max(states, axis=0)
    means = np.mean(states, axis=0)
    stds = np.std(states, axis=0)
    
    df_stats = pd.DataFrame({
        "Dimension": range(states.shape[1]),
        "Min": mins,
        "Max": maxs,
        "Mean": means,
        "Std": stds
    })
    
    print(df_stats.to_string(index=False))
    return states

if __name__ == "__main__":
    # 1. CartPole 的状态分布
    # 维度: [位置 x, 速度 x_dot, 角度 theta, 角速度 theta_dot]
    analyze_initial_states("CartPole-v1")
    
    # 2. LunarLander 的状态分布
    # 维度: [x, y, v_x, v_y, angle, angular_velocity, left_leg, right_leg]
    analyze_initial_states("LunarLander-v3") # 如果你用的是旧版 gym，可能是 v2