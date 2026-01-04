import subprocess
import os
import time
SCRIPT_NAME = "ga_acrobot.py"
BASE_ARGS = [
    "--rpc-port", "6000",
    "--authkey", "acrobot-rpc",
    "--processes", "16",
    "--population", "50",
    "--generations", "100",
    "--pool-size", "100",
    "--subset-k", "5",
    "--seed-refresh-frac", "0.2",
    "--seed-refresh-direction", "bottom",
    "--max-seed-age", "10",
    "--elite-frac", "0.2"
]

SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

OUT_DIR = "batch_results/acrobot"

def run_batch():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Starting Acrobot Batch Experiment (10 Runs)")
    print(f"Results will be saved to: {os.path.abspath(OUT_DIR)}")
    print("="*60)

    for i, seed in enumerate(SEEDS):
        print(f"\n[Run {i+1}/10] Starting with Seed {seed}...")
        start_time = time.time()
        cmd = ["python", SCRIPT_NAME] + BASE_ARGS
        cmd += ["--global-seed", str(seed)]
        cmd += ["--pool-rng-seed", str(seed)]
        cmd += ["--outdir", OUT_DIR]
        
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            print(f"[Run {i+1}/10] Finished Seed {seed} in {elapsed:.1f}s")
            
        except subprocess.CalledProcessError as e:
            print(f"[Run {i+1}/10] Failed Seed {seed}. Error: {e}")
            
    print("\n" + "="*60)
    print("All batch runs completed!")

if __name__ == "__main__":
    run_batch()