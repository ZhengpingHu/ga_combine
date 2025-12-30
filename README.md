# Visual-Neuro-Evolution-Control

A comprehensive framework combining state-of-the-art Computer Vision (YOLOv11) with Evolutionary Algorithms (GA/NEAT) to solve classic Reinforcement Learning control tasks. This project demonstrates an end-to-end pipeline where agents perceive the environment solely through visual inputs (pixel-level) and evolve control strategies without direct access to the environment's internal state vector.

## Hardware & Environment

The project was developed and tested on the following high-performance configurations:

* **Primary Workstation:** AMD Ryzen 3900X | NVIDIA RTX 2080Ti | 32GB RAM
* **Secondary Configuration:** Intel i7-12700KF | NVIDIA RTX 3070 | 32GB RAM
* **Language:** Python 3.9
* **Acceleration:** CUDA-enabled GPU acceleration for visual inference (YOLO/Pose models).

## System Architecture

The core pipeline follows a strictly visual-feedback loop:
`Gym Render Image` -> `Visual Perception Module` -> `State Vector` -> `Neural Controller` -> `Action`

### 1. Visual Perception Module (The "Eyes")
Unlike standard RL implementations that use internal state vectors (positions, velocities), this project extracts information directly from rendered frames using customized computer vision models based on **YOLOv11**.

#### Strategy by Environment:
* **Complex Orientation (LunarLander):** A robust **Two-Stage Recognition System** is implemented to handle severe jitter and angular flipping issues:
    * **Stage 1:** YOLOv11n-OBB (Oriented Bounding Box) detects the lander body and initial angle.
    * **Stage 2:** A crop of the detection area is passed to a custom **Pose Estimation Model** to precisely locate keypoints, significantly reducing sensor noise and orientation errors.
* **Multi-Component Systems (Acrobot, CartPole):** Detects distinct mechanical parts (e.g., poles, joints, cart bodies) to reconstruct the physical state.
* **Single Object Systems (MountainCar, Pendulum):** Specialized models to track the specific agent (Car or Pendulum rod).
    * *Note: Continuous MountainCar and Discrete MountainCar share the same unified vision model.*

### 2. Evolutionary Decision Module (The "Brain")
The control logic is derived through Genetic Algorithms rather than gradient-based RL methods (PPO/DQN).
* **Input:** Coordinates and angles derived from the visual module.
* **Controller Architecture:** Three distinct topologies were experimented with:
    1.  **NEAT (NeuroEvolution of Augmenting Topologies):** Evolving both weights and structures.
    2.  **Fixed-Topology Neural Networks:** Standard multi-layer perceptrons with evolved weights.
    3.  **Connection-Free Networks:** Simplified linear controllers or single-layer perceptions for specific tasks.

# Project 1: LunarLander-v3 (Visual-Evolutionary Control)

This module implements an end-to-end visual control system for `LunarLander-v3`. Unlike standard RL approaches that use internal state vectors, this agent perceives the environment strictly through rendered pixels, overcoming sensor noise, occlusion, and jitter using a custom evolutionary pipeline.

## Visual Perception Pipeline
The vision system runs as an independent RPC service (`yolo_rpc_server.py`), decoupling heavy inference from the simulation loop. This allows for high-throughput training on multi-GPU setups.

### 1. Two-Stage Recognition System
To handle the lander's symmetry and severe angle flipping issues during rotation, a robust fusion strategy is applied:
* **Stage 1: YOLOv11n-OBB** (Oriented Bounding Box)
    * Detects the lander body to provide the coarse location and initial angle.
* **Stage 2: YOLOv11-Pose** (Keypoint Estimation)
    * Identifies structural keypoints on the lander to refine the orientation precision.
* **Fusion Logic (Anti-Jitter):**
    * A **Hard Gate** threshold of `20.0°` is enforced. If the discrepancy between OBB and Pose angles exceeds this limit, the system falls back to OBB to prevent tracking loss.

### 2. Custom 7D State Vector
The agent does not access the game engine's physics. Instead, a state vector is reconstructed manually via frame differencing:
$$S = [x, y, \theta, \dot{x}, \dot{y}, \dot{\theta}, V_{total}]$$
> **Note:** Velocity is calculated based on `(current_frame - last_frame) / dt`, introducing realistic sensor delay and noise that the agent must learn to tolerate.

---

## Evolutionary Strategies

Two distinct evolutionary approaches were evaluated across 16 major iterations.

### Strategy A: Fixed-Topology MLP (Recommended)
*Ref: Experiment Run #13 (`final_run_with_smoothing`)*

Uses a fixed neural architecture evolved via a Genetic Algorithm with **Competitive Fitness Sharing**.

* **Architecture:** Fully Connected `7 (Input) -> 64 (Hidden) -> 4 (Output)`
* **Key Mechanisms:**
    * **Competitive Fitness:** Rewards unique solutions. If a seed is solved by many agents, its value decreases ($Fitness \propto 1/N_{solvers}$).
    * **Fitness Smoothing:** Agent scores are averaged over the last 5 generations to reduce variance.
    * **Aggressive Elitism:** Top 25% of agents are preserved.

### Strategy B: NEAT (NeuroEvolution)
*Ref: Experiment Run #16 (`neat_full_run`)*

Uses **NEAT** to evolve both weights and topology simultaneously, starting from a minimal structure.
* **Configuration:** [View neat.cfg](./neat.cfg)
* **Adaptation:** Allowed a higher "Max Seed Age" (40 gens) to give complex topologies time to mature.

---

## Training Configuration (Dynamic Seed Portfolio)

To prevent overfitting to specific terrain maps, a **Dynamic Seed Portfolio** system (`SeedPortfolioManager`) is implemented.

| Feature | Description |
| :--- | :--- |
| **Master Pool** | **256** pre-generated random maps (seeds). |
| **Active Subset** | Only **5** seeds are active per generation for evaluation efficiency. |
| **Dual-Refresh Strategy** | The active subset is updated based on two triggers:<br>1. **Age:** Seeds older than **5 generations** are discarded.<br>2. **Difficulty:** The **bottom 40%** (hardest) or top 40% (easiest) seeds are swapped out to maintain curriculum difficulty. |

### Reproduction Command (Best Fixed-Topology Run)
To reproduce the best performing agent (Run #13), use the following commands:

**1. Start Vision Server:**
```bash
python yolo_rpc_server.py --obb-model ./models/lander.pt --pose-model ./models/lander-pose.pt --device cuda:0 --gate-hard-deg 20 --port 6001

python gp_ga_7d_rpc.py \
    --rpc-port 6001 \
    --processes 16 \
    --population 120 \
    --generations 200 \
    --sigma 0.1 \
    --outdir runs_ga_final \
    --tag final_run_with_smoothing \
    --pool-size 256 \
    --subset-k 5 \
    --success-threshold 100.0 \
    --elite-frac 0.25 \
    --seed-refresh-frac 0.4 \
    --seed-refresh-direction bottom \
    --max-seed-age 5
```

## Iteration History & Structure

The directory structure reflects the chronological evolution of the algorithms, documenting the shift from a basic baseline to the final robust version.

```text
├── 07_obb_pose_7d/                 
│   └── Baseline: Simple GA with single-seed training.
├── 09_obb_pose_7d_5seed/           
│   └── Milestone: Introduction of the Seed Pool (64 seeds).
├── 10_obb_pose_7d_fitness_sharing/ 
│   └── Milestone: Introduction of Competitive Fitness mechanism.
├── 13_obb_pose_7d_fs_k_gen_cor/    
│   └── [Best Result] Added Fitness Smoothing (5-gen avg) & High Elitism.
└── 16_obb_pose_7d_NEAT_3/          
    └── Experimental: Long-run NEAT implementation.
```