import os
import json
import csv
import torch
import numpy as np

class ExperimentLogger:
    def __init__(self, args):
        self.run_name = f"seed_{args.global_seed}"
        self.base_dir = os.path.join(args.outdir, self.run_name)
        self.ckpt_dir = os.path.join(self.base_dir, "checkpoints")
        
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        config_path = os.path.join(self.base_dir, "config.json")
        with open(config_path, 'w') as f:
            safe_args = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list))}
            json.dump(safe_args, f, indent=4)

        self.csv_path = os.path.join(self.base_dir, "metrics.csv")
        self.csv_headers = ["Generation", "Best_Reward", "Avg_Reward", "Std_Reward", "Worst_Reward"]

        self.global_best_reward = -float('inf')

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)

        print(f"[Logger] Experiment initialized at: {self.base_dir}")

    def log_generation(self, generation, fitness_scores, best_agent_model, total_generations):
        scores = np.array(fitness_scores)
        best_r = np.max(scores)
        avg_r = np.mean(scores)
        std_r = np.std(scores)
        worst_r = np.min(scores)

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([generation, best_r, avg_r, std_r, worst_r])

        if best_r >= self.global_best_reward:
            self.global_best_reward = best_r
            torch.save(best_agent_model.state_dict(), os.path.join(self.base_dir, "best_ever.pt"))

        if generation % 10 == 0 or generation == 0:
            ckpt_name = f"gen_{generation:03d}.pt"
            torch.save(best_agent_model.state_dict(), os.path.join(self.ckpt_dir, ckpt_name))

        if generation == total_generations - 1:
            torch.save(best_agent_model.state_dict(), os.path.join(self.base_dir, "final.pt"))