#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
from tqdm import tqdm
from power_logger import SystemPowerLogger

def main():
    parser = argparse.ArgumentParser(description="3-Minute System Power Monitor")
    parser.add_argument("--out", type=str, default="power_log.csv", help="Output CSV filename")
    parser.add_argument("--duration", type=int, default=180, help="Duration to record in seconds (default: 180)")
    args = parser.parse_args()

    print(f"==================================================")
    print(f"[*] 准备开始硬件能耗监控...")
    print(f"[*] 记录时长: {args.duration} 秒")
    print(f"[*] 输出文件: {args.out}")
    print(f"==================================================")
    print(f"[!] 请在倒计时结束后，立即在另一个终端启动你的训练脚本！")
    
    # 给你 3 秒钟的准备时间去切换窗口
    for i in range(3, 0, -1):
        print(f"准备... {i}")
        time.sleep(1)

    # 1. 实例化并启动我们在后台多线程运行的记录器
    logger = SystemPowerLogger(log_file=args.out, interval=0.5)
    logger.start()

    # 2. 带有进度条的精准 3 分钟延时阻塞
    print("\n[*] 监控中... (请保持训练脚本运行)")
    try:
        # 使用 tqdm 画一个漂亮的进度条，让你直观看到 3 分钟还剩多久
        for _ in tqdm(range(args.duration), desc="Recording Power", unit="sec"):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] 监控被用户手动提前终止。")

    # 3. 时间到，停止记录并保存
    logger.stop()
    print(f"\n[*] 监控结束！数据已完美写入: {args.out}")
    print(f"==================================================")

if __name__ == "__main__":
    main()