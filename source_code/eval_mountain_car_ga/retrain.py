#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ultralytics import YOLO

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "mountaincar_noisy.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"[!] 错误: 找不到配置文件 {yaml_path}")
        return
        
    # [请确认预训练模型存在]
    pretrained_model = os.path.join(current_dir, "best.pt")
    if not os.path.exists(pretrained_model):
        print(f"[!] 错误: 找不到 Clean 预训练模型 {pretrained_model}")
        return

    print(f"[*] 开始加载 Clean 环境预训练权重: {pretrained_model}")
    model = YOLO(pretrained_model) 

    print(f"[*] 使用配置文件: {yaml_path}")
    print("[*] 开始在 Noisy 数据集上进行网络微调 (Fine-tuning)...")
    
    results = model.train(
        data=yaml_path,        
        epochs=50,             # 微调通常 50 epoch 内就能收敛      
        imgsz=640,             # YOLO 通常使用 640 尺寸训练 (它会自动 padding 600x400 的图像)
        batch=16,              
        device="0",            
        project="YOLO_MountainCar",
        name="finetuned_noisy",
        patience=15,           
        lr0=0.001              # 使用较低的学习率以保护预训练特征
    )
    
    print("[*] 微调完成！")
    print("新模型的权重保存在: YOLO_MountainCar/finetuned_noisy/weights/best.pt")
    print("请将其重命名为 best_noise.pt，随后放入你的 GA 评估流程中！")

if __name__ == "__main__":
    main()