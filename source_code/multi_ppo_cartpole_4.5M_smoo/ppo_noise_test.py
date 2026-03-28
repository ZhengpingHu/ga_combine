import os
import gymnasium as gym
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium import spaces

# ==========================================
# 1. 环境扰动 Wrapper (注入噪声与杂乱背景)
# ==========================================
class VisuallyClutteredWrapper(gym.Wrapper):
    def __init__(self, env, gaussian_std=25.0, add_clutter=True):
        super().__init__(env)
        self.gaussian_std = gaussian_std
        self.add_clutter = add_clutter
        self.clutter_texture = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)

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
# 2. 像素化 Wrapper (适配 PPO 的 CNN 输入)
# ==========================================
class PixelCartPole(gym.ObservationWrapper):
    def __init__(self, env, img_size=(84, 84)):
        super().__init__(env)
        self.img_size = img_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.uint8
        )

    def observation(self, obs):
        img = self.env.render()
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        return img

# ==========================================
# 3. 环境构建工厂
# ==========================================
def make_noisy_pixel_env(env_id="CartPole-v1", seed=42):
    def _init():
        base_env = gym.make(env_id, render_mode="rgb_array")
        # 1. 注入噪声
        noisy_env = VisuallyClutteredWrapper(base_env, gaussian_std=25.0, add_clutter=True)
        # 2. 转换为 84x84 像素输入供 PPO 使用
        pixel_env = PixelCartPole(noisy_env)
        pixel_env.action_space.seed(seed)
        return pixel_env
    return _init

# ==========================================
# 4. 测试逻辑
# ==========================================
def test_zeroshot(model_path, env):
    print(f"\n[*] 正在加载干净环境中训练的 PPO 模型: {model_path} ...")
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"[!] 加载失败: {e} (请确保路径正确)")
        return

    print("[*] 开始进行 Zero-Shot 噪声环境测试 (10 局)...")
    total_rewards = []
    for ep in range(10):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0] # VecEnv returns array
        total_rewards.append(ep_reward)
        print(f"    - 局数 {ep+1}: 得分 {ep_reward:.1f}")
    
    print(f"[*] Zero-Shot 平均得分: {np.mean(total_rewards):.1f} / 500.0")

def train_from_scratch(env, total_timesteps=200_000):
    print(f"\n[*] 开始在高度噪声环境中从头训练 PPO ({total_timesteps} 步) ...")
    model = PPO("CnnPolicy", env, verbose=1, device="cuda", seed=101, n_steps=512, batch_size=256)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    save_path = "ppo_noisy_cartpole_test"
    model.save(save_path)
    print(f"[*] 训练完成，模型已保存至 {save_path}.zip")

if __name__ == "__main__":
    # 创建带噪声的向量化环境 (4帧堆叠)
    vec_env = make_vec_env(
        env_id="CartPole-v1", 
        n_envs=1, # 测试和简单训练用单线程看日志更清晰
        env_kwargs={"render_mode": "rgb_array"},
        wrapper_class=lambda e: PixelCartPole(VisuallyClutteredWrapper(e, gaussian_std=25.0, add_clutter=True))
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # 1. 测试你的旧模型 (请把 "final_model.zip" 替换为你实际的 PPO 模型文件路径)
    old_model_path = "./final_ppo_model" 
    if os.path.exists(f"{old_model_path}.zip"):
        test_zeroshot(old_model_path, vec_env)
    else:
        print(f"\n[!] 未找到 {old_model_path}.zip，跳过 Zero-shot 测试。")

    # 2. 从头训练 PPO
    # (这里先跑 20 万步看看趋势，如果一直卡在 10-20 分不涨，论文里的论点就成了)
    train_from_scratch(vec_env, total_timesteps=200_000)
    
    vec_env.close()