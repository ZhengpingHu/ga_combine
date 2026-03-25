#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import csv
import threading
import psutil

try:
    import pynvml
except ImportError:
    raise RuntimeError("Please install pynvml: pip install pynvml")

class SystemPowerLogger:
    def __init__(self, log_file="power_log.csv", interval=0.5, gpu_index=0):
        """
        Initializes the background power logger for CPU and GPU.
        
        Args:
            log_file: Path to save the CSV log.
            interval: Sampling interval in seconds.
            gpu_index: Index of the NVIDIA GPU to monitor (usually 0).
        """
        self.log_file = log_file
        self.interval = interval
        self.gpu_index = gpu_index
        self.is_running = False
        self.logger_thread = None
        
        # Initialize NVIDIA Management Library
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
        print(f"[*] PowerLogger initialized. Monitoring GPU: {gpu_name}")

    def _log_loop(self):
        """
        The core loop running in a background thread to fetch and write data.
        """
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Write CSV header
            writer.writerow(["Timestamp_sec", "CPU_Usage_Percent", "GPU_Power_Watts"])
            
            start_time = time.time()
            
            while self.is_running:
                # 1. Get Elapsed Time
                elapsed = time.time() - start_time
                
                # 2. Get CPU Usage (%)
                cpu_pct = psutil.cpu_percent(interval=None)
                
                # 3. Get GPU Power (returned in milliwatts, convert to Watts)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                gpu_power_w = power_mw / 1000.0
                
                # Write to CSV
                writer.writerow([f"{elapsed:.2f}", f"{cpu_pct:.1f}", f"{gpu_power_w:.2f}"])
                
                # Sleep for the specified interval
                time.sleep(self.interval)

    def start(self):
        """
        Starts the background logging thread.
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.logger_thread = threading.Thread(target=self._log_loop, daemon=True)
        self.logger_thread.start()
        print(f"[*] Power logging started -> saving to {self.log_file}")

    def stop(self):
        """
        Stops the logging thread and cleans up NVML.
        """
        if not self.is_running:
            return
            
        self.is_running = False
        if self.logger_thread:
            self.logger_thread.join()
            
        pynvml.nvmlShutdown()
        print("[*] Power logging stopped.")

# Quick test execution
if __name__ == "__main__":
    logger = SystemPowerLogger(log_file="test_power.csv", interval=0.5)
    logger.start()
    
    print("Simulating system load for 5 seconds...")
    # Put some dummy load here just to test
    for _ in range(5):
        time.sleep(1)
        
    logger.stop()
    print("Test complete. Check test_power.csv!")