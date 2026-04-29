# LunarLander DQN Agent

A Deep Q-Network (DQN) implementation for training an agent to land a spacecraft safely in the LunarLander-v3 environment from Gymnasium.

## 📋 Project Overview

This project implements a complete reinforcement learning pipeline for the LunarLander-v3 environment, including:

- **DQN Agent**: A deep neural network-based Q-learning agent with experience replay
- **Environment Integration**: Full integration with Gymnasium's LunarLander-v3 environment
- **Visualization & Analysis**: Tools to visualize episode data and analyze agent behavior
- **Demo System**: Ready-to-run demos to test the agent and environment

### Goal

Train an agent to safely land a lunar module between two landing pad flags with minimal fuel consumption and stable balance.

---

## 🎮 Environment Details

### State Space (8 values)
The agent observes:
- **x position**: Horizontal position relative to landing pad
- **y position**: Vertical position (decreases as it falls)
- **x velocity**: Horizontal speed
- **y velocity**: Vertical speed (negative when falling)
- **angle**: Tilt angle of the lander
- **angular velocity**: Rate of rotation
- **left leg contact**: Binary flag for left leg touching ground
- **right leg contact**: Binary flag for right leg touching ground

### Action Space (4 actions)
The agent can take one of four actions each timestep:
- **0**: Do nothing
- **1**: Fire left orientation engine
- **2**: Fire main engine
- **3**: Fire right orientation engine

### Reward Function
The agent receives rewards for:
- ✅ Moving closer to the landing pad
- ✅ Landing safely (legs touching ground)
- ✅ Staying upright (low angle)
- ✅ Reducing speed
- ❌ Penalties for crashing, tilting badly, wasting fuel, and inefficient movement

---

## 🏗️ Project Structure

```
gdsc-ml2-1/
├── main.py                           # Entry point with three modes
├── src/
│   ├── dqn_agent.py                 # DQN agent implementation
│   ├── dqn_lunarlander.pth          # Trained model weights
│   ├── env/
│   │   ├── env_info.py              # Environment constants & info printer
│   │   └── environment_guide.md     # Detailed environment documentation
│   └── visualization/
│       └── plot_states.py           # Data collection & plotting utilities
├── demos/
│   ├── demo_env.py                  # Demo runner for the environment
│   └── requirements.txt             # Python dependencies
└── README.md                        # This file
```

---

## 🤖 Core Components

### 1. **DQN Agent** (`src/dqn_agent.py`)

Implements the Deep Q-Network algorithm with:

**QNetwork**: Neural network that maps state (8 dims) → Q-values (4 actions)
- Input layer: 8 neurons (state)
- Hidden layer 1: 128 neurons + ReLU activation
- Hidden layer 2: 128 neurons + ReLU activation
- Output layer: 4 neurons (one Q-value per action)

**ReplayBuffer**: Stores and samples past experiences to break temporal correlations
- Capacity: 50,000 experiences
- Sampling: Random batches for training stability

**DQNAgent**: Main training logic
- Epsilon-greedy exploration: Starts 100% random, decays to 1% random over 10,000 steps
- Two networks: Online (trained) and Target (frozen, updated every 500 steps)
- Discount factor (γ): 0.99 (future rewards matter)
- Learning rate: 0.001 (Adam optimizer)
- Batch size: 64 experiences per training step

### 2. **Environment Module** (`src/env/`)

**env_info.py**: Easy access to environment constants
- `STATE_NAMES`: List of all state variable names
- `ACTION_MEANINGS`: Mapping of action indices to descriptions
- `print_env_info()`: Prints formatted environment information

### 3. **Visualization** (`src/visualization/plot_states.py`)

**collect_episode_data()**: Runs an episode and records all state/reward data
**plot_main_states()**: Visualizes position and velocity over time
**plot_angle_states()**: Visualizes angle and angular velocity
**plot_leg_contacts()**: Visualizes leg contact with ground
**plot_reward_curve()**: Plots reward per step with moving average smoothing
**moving_average()**: Utility for smoothing noisy signals

### 4. **Demo** (`demos/demo_env.py`)

Runs 3 episodes of the environment with random actions (baseline behavior).

---

## 🚀 Usage

### Prerequisites

Python 3.11+ with a virtual environment set up containing all dependencies from `demos/requirements.txt`:
- gymnasium (RL environment)
- torch (neural network framework)
- numpy (numerical computing)
- matplotlib (visualization)
- pygame (rendering)
- Box2D (physics simulation)

### Running the Project

The `main.py` script provides three modes:

#### 1. **Demo Mode** (Default)
Watch 3 random episodes with visualization:
```bash
python main.py --mode demo
```
Shows the agent taking random actions. Total reward should be negative (untrained agent).

#### 2. **Plotting Mode**
Run one episode and visualize the data:
```bash
python main.py --mode plot
```
Generates four plots:
- Position & Velocity vs. Time
- Angle & Angular Velocity vs. Time
- Leg Contact vs. Time
- Reward per Step (with moving average smoothing)

#### 3. **Info Mode**
Print environment information:
```bash
python main.py --mode info
```
Displays:
- All 8 state variable names and indices
- All 4 action meanings

---

## 📊 Training Notes

Currently, the agent uses random actions (untrained). The trained model weights are stored in `src/dqn_lunarlander.pth`.

### Key Observations from Random Policy

- **Total Reward**: Negative (agent crashes/wastes fuel)
- **Angle**: Increases over time → lander tilts and loses balance
- **Angular Velocity**: Builds up → uncontrolled spinning
- **Vertical Velocity**: Becomes increasingly negative → crashes into ground
- **Leg Contacts**: Usually don't occur before crash (unstable descent)

### What the Agent Needs to Learn

1. **Controlled Descent**: Balance vertical velocity to avoid crashing
2. **Sideways Stability**: Manage x velocity and angle to stay upright
3. **Fuel Efficiency**: Use engines sparingly
4. **Landing**: Time the final descent to land with legs touching

---

## 🛠️ Dependencies

All dependencies are listed in `demos/requirements.txt`:

```
gymnasium==1.2.3          # RL environments
torch==2.11.0             # Neural networks
numpy==2.4.4              # Numerical computing
matplotlib==3.10.8        # Plotting
pygame==2.6.1             # Rendering
Box2D==2.3.10             # Physics simulation
cloudpickle==3.1.2        # Serialization
... (and others)
```

---

## 📝 Algorithm Details

### Deep Q-Learning Formula

The agent learns to estimate Q-values (expected future reward for each action):

$$Q(s, a) \approx r + \gamma \max_{a'} Q(s', a')$$

Where:
- $s$ = current state
- $a$ = action taken
- $r$ = immediate reward
- $\gamma$ = discount factor (0.99)
- $s'$ = next state
- $a'$ = best action in next state

### Target vs. Online Network

- **Online Network**: Updated every step with gradient descent
- **Target Network**: Frozen copy updated every 500 steps
  - **Purpose**: Provides stable training targets, reduces oscillations

### Experience Replay

- Stores up to 50,000 transitions (state, action, reward, next_state, done)
- Samples random batches of 64 during training
- **Benefits**: Breaks temporal correlations, improves sample efficiency

### Exploration Strategy (Epsilon-Greedy)

- **ε-start**: 1.0 (100% random exploration)
- **ε-end**: 0.01 (1% exploration, 99% greedy)
- **ε-decay**: Linear over 10,000 steps

---

## 🔍 Example Output

### Demo Output:
```
Episode 1
Total Reward: -245.32

Episode 2
Total Reward: -312.18

Episode 3
Total Reward: -289.45
```

### Info Output:
```
=== LunarLander Environment Info ===

State Variables:
0: x_position
1: y_position
2: x_velocity
3: y_velocity
4: angle
5: angular_velocity
6: left_leg_contact
7: right_leg_contact

Actions:
0: do nothing
1: fire left orientation engine
2: fire main engine
3: fire right orientation engine
```

---

## 📚 Further Reading

- **Gymnasium Docs**: https://gymnasium.farama.org/
- **DQN Paper**: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- **LunarLander Details**: https://gymnasium.farama.org/environments/box2d/lunar_lander/

---

## 📧 Notes

- The project uses PyTorch on GPU if available, otherwise CPU (auto-detected)
- Pygame is used for rendering environments
- Box2D provides realistic physics simulation for the lunar lander
- Currently set up for single-episode data collection; can be extended for training loops

