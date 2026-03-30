from ultralytics import YOLO

def main():
    print("[*] 开始加载 Clean 环境下训练好的 best.pt...")
    # 加载你之前训练好的模型作为预训练权重
    model = YOLO("best.pt") 

    print("[*] 开始在 Noisy 数据集上进行微调...")
    # 开始微调训练
    results = model.train(
        data="acrobot_noisy.yaml",
        epochs=50,             # 微调不需要太多 epoch，50-100 足够
        imgsz=500,             # Acrobot 画面大小
        batch=16,              # 视你的显存大小调整 (16 或 32)
        device="0",            # 使用第一块 GPU
        project="YOLO_Acrobot",
        name="finetuned_noisy",
        patience=20,           # 如果验证集 mAP 连续 20 轮不上升则提前停止
        lr0=0.001              # 使用比从头训练稍低的学习率，防止破坏已有结构
    )
    
    print("[*] 微调完成！")
    print("新模型的权重保存在: YOLO_Acrobot/finetuned_noisy/weights/best.pt")
    print("你可以将它重命名为 best_noise.pt 以供验证脚本使用！")

if __name__ == "__main__":
    main()