# Visual-Genetic-Algorithm-Control-Agent

A comprehensive framework combining Computer Vision (YOLOv11) with Evolutionary Algorithms (GA/NEAT) to solve classic Reinforcement Learning control tasks. This project demonstrates an end-to-end pipeline where agents perceive the environment solely through visual inputs (pixel-level) and evolve control strategies without direct access to the environment's internal state vector.

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



# Project 2: Classical Control - Acrobot-v1

The Acrobot task involves swinging up a two-link robot arm to a given height. This module demonstrates that with precise visual feature extraction, a simple **Linear Controller (Connection-Free)** is sufficient to solve the task, eliminating the need for deep neural networks.

## Visual Perception (Pose-Only)
Unlike LunarLander, the Classical Control suite relies exclusively on **YOLOv11-Pose** for simplified inference and higher throughput, as the environments have cleaner backgrounds and distinct structural components.

* **Model:** `YOLOv11-Pose`
* **Keypoints Detected (3 Total):**
    1.  **Base:** The fixed anchor point.
    2.  **Joint:** The connection between the two links.
    3.  **Tip:** The free end of the second link.
* **State Vector Construction (8D):**
    The raw keypoints are converted into a relative coordinate system to form the input vector:
    $$S = [X_{joint}, Y_{joint}, X_{tip}, Y_{tip}, \dot{X}_{joint}, \dot{Y}_{joint}, \dot{X}_{tip}, \dot{Y}_{tip}]$$
    * *Coordinates are relative to the Base point.*
    * *Velocities are calculated via temporal differencing.*

## Controller & Evolutionary Strategy

### Architecture: Linear Policy (No Hidden Layers)
Since the state features (positions + velocities) are highly descriptive, a single transformation layer is used.
* **Type:** `nn.Linear(8, 3, bias=True)`
* **Structure:** `8 Inputs -> 3 Outputs` (Softmax logic implied for discrete actions: Apply Torque -1, 0, +1).
* **Parameter Count:** Only **27** parameters (Weights + Bias).

### GA Configuration (Dynamic Seed Pool)
Even for this simpler task, the **Dual-Refresh Seed Pool** strategy is employed to ensure the agent doesn't memorize a specific starting momentum or gravity variation.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Population** | 50 | Smaller population sufficient for linear search space. |
| **Generations** | 100 | Fast convergence expected. |
| **Seed Pool Size** | 100 | Total unique starting conditions. |
| **Subset Size** | 5 | Seeds evaluated per generation. |
| **Refresh Strategy** | Bottom 20% | Replaces the 20% hardest seeds and seeds older than 10 gens. |
| **Elite Fraction** | 0.2 | Top 20% preserved. |

## Reproduction Commands

**1. Start Vision Server (Pose Mode):**
```bash
python acrobot_server.py --model ./models/acrobot_pose.pt --port 6000
```

**2. Start Training Client:**
```bash
python gp_ga_acrobot.py \
    --rpc-port 6000 \
    --authkey acrobot-rpc \
    --processes 16 \
    --population 50 \
    --generations 100 \
    --pool-size 100 \
    --subset-k 5 \
    --seed-refresh-frac 0.2 \
    --seed-refresh-direction bottom \
    --max-seed-age 10 \
    --outdir runs_acrobot
```


# Project 3: Classical Control - CartPole-v1

The "Hello World" of Reinforcement Learning. The goal is to balance a pole attached to a moving cart. This implementation proves that visual servoing using a simple linear matrix multiplication is sufficient to maintain equilibrium, provided the visual state estimation is low-latency.

## Visual Perception (Pose-Only)
The vision system tracks the mechanical structure to reconstruct the standard 4-variable state vector used in control theory.

* **Model:** `YOLOv11-Pose`
* **Keypoints Detected (2 Total):**
    1.  **Base:** The center pivot point of the cart.
    2.  **Tip:** The top end of the pole.
* **Calibration:**
    * A specific **Gain Factor** (`1.12`) is applied during inference. This likely compensates for visual perspective distortion or acts as a proportional control gain multiplier to make the agent more responsive to small angular deviations.
* **State Vector (4D):**
    $$S = [x_{cart}, \dot{x}_{cart}, \theta_{pole}, \dot{\theta}_{pole}]$$
    * Derived from the raw pixel coordinates of the Base and Tip.

## Controller & Evolutionary Strategy

### Architecture: Linear Policy (No Hidden Layers)
Similar to Acrobot, the problem space is linearly separable regarding the stabilization task.
* **Type:** `nn.Linear(4, 2, bias=True)`
* **Structure:** `4 Inputs -> 2 Outputs` (Left / Right).
* **Complexity:** Minimalist design with only **10** parameters.

### GA Configuration
The training utilizes the standard Dynamic Seed Portfolio to ensure the agent can recover from various initial random perturbations.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Population** | 50 | Small population size. |
| **Generations** | 20 | Rapid convergence allows for short training sessions. |
| **Seed Pool Size** | 100 | Number of unique starting states. |
| **Subset Size** | 5 | Seeds evaluated per generation (High throughput). |
| **Refresh Rate** | 40% (0.4) | Aggressive refresh strategy to prevent overfitting. |
| **Elite Fraction** | 0.2 | Top 20% preserved. |

## Reproduction Commands

**1. Start Vision Server (With Gain Calibration):**
*Note the specific gain parameter used for this environment.*
```bash
python yolo_server.py --model ./models/best.pt --port 6000 --gain 1.12
```

**2. Start Training Client:**
```bash
python ga_client.py \
    --rpc-port 6000 \
    --authkey cartpole-rpc \
    --processes 16 \
    --population 50 \
    --generations 20 \
    --pool-size 100 \
    --subset-k 5
```



# Project 4: MountainCar-v0 (Discrete)

The MountainCar environment presents a classic exploration challenge: an underpowered car must drive up a steep hill. Since gravity is stronger than the engine, the agent must learn to move *away* from the goal (to the left) to build momentum—a counter-intuitive strategy that linear controllers often fail to master without feature engineering.

## Visual Perception (Unified Model)
Both discrete and continuous versions of MountainCar share the same vision backend.
* **Model:** `YOLOv11-Pose`
* **Object Detected:** The Car Body.
* **State Vector (2D):**
    $$S = [x_{car}, \dot{x}_{car}]$$
    * Position is derived from the bounding box center.
    * Velocity is calculated via frame differencing.

## Controller Architecture Comparison

This project conducted a comparative study between two architectures to demonstrate the necessity of non-linearity.

### Experiment A: Linear Policy (Failed)
* **Structure:** `nn.Linear(2, 3)` (No hidden layers).
* **Outcome:** **Failure.** The agent could not solve the task. A simple linear matrix multiplication is insufficient to represent the complex "swing-up" policy required to escape the valley from raw coordinates.

### Experiment B: Shallow MLP (Success)
* **Structure:** `nn.Linear(2, 8) -> Tanh -> nn.Linear(8, 3)`
* **Hidden Layer:** Single layer with **8 neurons** and **Tanh** activation.
* **Outcome:** **Success.** The introduction of a small hidden layer provided enough non-linearity for the agent to learn the energy-pumping strategy.

## GA Configuration (Successful Run)

Training required a high elite fraction to preserve the specific rhythmic behavior once discovered.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Population** | 50 | - |
| **Generations** | 100 | - |
| **Elite Fraction** | 0.4 (Top 40%) | Very high elitism to protect successful traits. |
| **Seed Refresh** | Bottom 25% | Replaces the worst performing seeds. |
| **Network** | MLP (8 Nodes) | **Required for convergence.** |

## Reproduction Commands

To reproduce the successful MLP experiment:

**1. Start Vision Server:**
```bash
python yolo_server_mc.py --model ./models/best.pt --port 6001
```

**2. Start Training Client (MLP Version):**
```bash
python ga_client_mc.py \
    --rpc-port 6001 \
    --population 50 \
    --generations 100 \
    --processes 16 \
    --pool-size 100 \
    --subset-k 5 \
    --seed-refresh-direction bottom \
    --elite-frac 0.4
```


# Project 5: MountainCarContinuous-v0 (Experimental Analysis)

This module explores the continuous control version of the MountainCar problem. Unlike the discrete version, the agent must output a continuous force value $[-1.0, 1.0]$. This project serves as a critical **failure analysis case study**, highlighting the challenges of sparse rewards in evolutionary reinforcement learning.

## Visual Perception
Shared infrastructure with the discrete version.
* **Model:** `YOLOv11-Pose`
* **State:** $$S = [x, \dot{x}]$$
* **Control Group:** To isolate visual estimation errors, experiments were also conducted using **Ground Truth** data directly from the environment.

## Architectures & Evolution

### 1. Linear Policy (Baseline)
* **Structure:** `Input(2) -> Linear(1) -> Tanh`
* **Outcome:** Failed. The linear mapping could not capture the swing-up dynamics.

### 2. MLP Policy (Enhanced)
* **Evolution:** Upgraded from 8 hidden nodes (insufficient) to **16 hidden nodes**.
* **Structure:** `Input(2) -> Linear(16) -> ReLU -> Linear(1) -> Tanh`
* **Outcome:** **Convergence to Local Minimum.**
    * The agent learned a "Stay Still" policy.
    * **Reasoning:** In `MountainCarContinuous`, applying force costs energy (negative reward). Since the goal reward (100) is hard to reach via random exploration, the GA optimized for the strategy that minimizes penalty: doing nothing.

## Reward Shaping: Energy-Based Fitness
To combat the "Stay Still" local minimum, a custom **Fitness Shaping** mechanism was implemented (applied only to internal fitness, not final reported reward) to encourage momentum.

**Logic:** If the agent fails to reach the goal, add a bonus based on the **Maximum Kinetic Energy** achieved.

```python
# Pseudo-code of the Shaping Logic implemented
if raw_reward <= 0.0:
    max_energy = calculate_max_energy(pos_history, vel_history)
    fitness_score = raw_reward + (max_energy * 10.0)
else:
    fitness_score = raw_reward
```
## GA Configuration (Continuous Specific)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Population**|50|-|
| **Generations** | 100 | - |
| **Elite Fraction** | 0.4 | High elitism to preserve high-energy mutants. |
| **Network** | MLP (16 Nodes) | Upgraded from 8 nodes. |
| **Input Data** | Hybrid | Tested both YOLO-Inference and Ground-Truth.|

## Reproduction Commands
**1. Start Vision Server:**
```bash
python yolo_server_mc.py --model ./models/best.pt --port 6001
```

**2. Start Training Client (16-Node MLP):**

```bash
python ga_client_mc.py \
    --rpc-port 6001 \
    --population 50 \
    --generations 100 \
    --processes 16 \
    --pool-size 100 \
    --subset-k 5 \
    --seed-refresh-direction bottom \
    --elite-frac 0.4
```


# Project 6: Pendulum-v1 (Comparative Study)

The Pendulum task (swinging up and balancing a pendulum) requires precise continuous torque control $[-2.0, 2.0]$. This project serves as a crucial **ablation study**, comparing **Visual Perception** against **Direct State Access** to quantify the impact of sensor noise on stabilization.

## Experimental Setup (2x2 Factorial Design)

To isolate the causes of failure, four distinct configurations were tested:
1.  **Vision Input + Linear Policy**
2.  **Vision Input + MLP (16 Nodes)**
3.  **Direct Input + Linear Policy**
4.  **Direct Input + MLP (16 Nodes)**

### 1. State Representations
* **Visual State (YOLO):** $S = [Tip{x}, Tip{y}, \dot{x}{tip}, \dot{y}{tip}]$
    * Derived from YOLOv11 keypoint tracking. Velocities calculated via frame differencing.
* **Direct State (Ground Truth):** $S = [\cos\theta, \sin\theta, \dot{\theta}]$
    * **Raw 3D output** directly from the Gymnasium environment. No mapping or modification was applied.

### 2. Network Architectures
* **Linear (No Hidden):** `Input(4) -> Linear(1) -> Tanh * 2.0`
* **MLP (16 Nodes):** `Input(4) -> Linear(16) -> ReLU -> Linear(1) -> Tanh * 2.0`
    * *Note: The output is scaled by 2.0 to match the environment's torque limits.*

## Results & Analysis

| Input Source | Network Architecture | Outcome | Analysis |
| :--- | :--- | :--- | :--- |
| **Direct (GT)** | **MLP (16 Nodes)** | **Success** | The agent successfully swings up **and maintains stable balance** at the top. |
| **Direct (GT)** | Linear |  Failure | Can swing up but cannot stabilize. Lacks non-linearity for energy control. |
| **Vision (YOLO)**| MLP (16 Nodes) | Partial | Can swing up to the top, but **fails to stabilize**. Visual noise in velocity calculation prevents precise micro-adjustments. |
| **Vision (YOLO)**| Linear | Failure | Fails to swing up effectively. |

**Conclusion:** Solving Pendulum requires **both** a non-linear policy (to manage the swing-up energy) and high-fidelity state estimation (for the delicate equilibrium). Visual differentiation noise proved too high for the final stabilization phase.

## GA Configuration

| Parameter | Vision Mode | Direct Mode |
| :--- | :--- | :--- |
| **Population** | 50 | 100 |
| **Generations** | 100 | 100 |
| **Seed Refresh** | 0.4 (40%) | 0.4 (40%) |
| **Subset Size** | 5 | 5 |

## Reproduction Commands

**Scenario A: The Winning Model (Direct + MLP)**
```bash
python ga_pendulum_direct.py \
    --population 100 \
    --generations 100 \
    --processes 16 \
    --sigma 0.1
```

**Scenario B: Visual Approach (Swings up, unstable)**
**1.Start Vision Server:**
```bash
python pendulum_server.py
```
**2.Start GA Client:**
```bash
python ga_pendulum.py \
    --population 50 \
    --generations 100 \
    --subset-k 5 \
    --seed-refresh-rate 0.4
```

# References:
[1] Ultralytics, "YOLO11," Ultralytics Documentation. [Online]. Available: https://docs.ultralytics.com/models/yolo11/. [Accessed: Dec. 30, 2025].

[2] Farama Foundation, "Gymnasium," Farama Foundation Documentation. [Online]. Available: https://gymnasium.farama.org/. [Accessed: Dec. 30, 2025].

[3] S. G. Ficici and J. B. Pollack, "Pareto optimality in coevolutionary learning," in Proceedings of the 6th European Conference on Advances in Artificial Life (ECAL), Prague, Czech Republic, 2001, pp. 316–325.
