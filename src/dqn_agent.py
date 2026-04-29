import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),  # 8 → 128
            nn.ReLU(),
            nn.Linear(hidden, hidden),     # 128 → 128
            nn.ReLU(),
            nn.Linear(hidden, action_dim)  # 128 → 4   (one Q-value per action)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        # deque automatically drops old experiences when full
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save one transition (one timestep of experience)."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Randomly pick a batch of past experiences to learn from."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),       # shape: (batch, 8)
            torch.LongTensor(actions),                 # shape: (batch,)
            torch.FloatTensor(rewards),                # shape: (batch,)
            torch.FloatTensor(np.array(next_states)),  # shape: (batch, 8)
            torch.FloatTensor(dones)                   # shape: (batch,)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network agent for LunarLander-v3.

    State  (input)  : 8 floats  →  [x, y, vx, vy, angle, ang_vel, leg_L, leg_R]
    Action (output) : int 0–3   →  do nothing / left / main / right engine
    """

    def __init__(
        self,
        state_dim:          int   = 8,
        action_dim:         int   = 4,
        lr:                 float = 1e-3,     # how fast the network learns
        gamma:              float = 0.99,     # how much future rewards matter
        batch_size:         int   = 64,       # experiences per training step
        buffer_size:        int   = 50_000,   # max experiences to remember
        target_update_freq: int   = 500,      # how often to sync target network
        eps_start:          float = 1.0,      # start fully random
        eps_end:            float = 0.01,     # end mostly greedy
        eps_decay_steps:    int   = 10_000,   # steps to go from start → end
        min_buffer:         int   = 1_000,    # don't train until buffer has this many
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer = min_buffer
        self.step_count = 0           # total timesteps so far

        # Epsilon (exploration rate) 
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = (eps_start - eps_end) / eps_decay_steps

        # Device: use GPU if available, otherwise CPU 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Two networks 
        # Online network : trained every step
        # Target network : frozen copy, updated every N steps
        self.online = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())  # start identical
        self.target.eval()  # target never collects gradients

        # Optimizer and memory 
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    #  select_action(state)
    #  Input : np.ndarray of shape (8,)
    #  Output: int in {0, 1, 2, 3}

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon  → pick a random action  (explore)
        With probability 1-epsilon → pick the best action  (exploit)

        Early in training epsilon is high so the agent explores a lot.
        Over time epsilon decays so the agent trusts its learned Q-values.
        """
        assert len(state) == self.state_dim, (
            f"Expected state of length {self.state_dim}, got {len(state)}"
        )

        if random.random() < self.eps:
            # Explore: random action, ignore state completely
            return random.randrange(self.action_dim)

        # Exploit: ask the neural network which action looks best
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # unsqueeze(0) turns shape (8,) → (1, 8) because the network
        # expects a batch dimension even when processing one state

        with torch.no_grad():  # no gradient needed, we're just predicting
            q_values = self.online(state_tensor)   # shape: (1, 4)

        return q_values.argmax(dim=1).item()       # index of highest Q-value

    #  learn(batch)
    #  Called every timestep once the buffer has enough experiences

    def learn(self, state, action, reward, next_state, done):
        """
        Store the latest transition, then run one training step.

        Args:
            state      : np.ndarray (8,)  — state before action
            action     : int              — action taken
            reward     : float            — reward received
            next_state : np.ndarray (8,)  — state after action
            done       : bool             — did the episode end?
        """
        # Step 1 — store this experience in memory
        self.buffer.push(state, action, reward, next_state, float(done))

        # Step 2 — only start training once we have enough memories
        if len(self.buffer) < self.min_buffer:
            return None

        # Step 3 — randomly sample a batch of past experiences
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Step 4 — compute what Q-values the online network currently predicts
        # .gather picks the Q-value for the specific action that was actually taken
        q_predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # shape: (batch,)  — one Q-value per experience in the batch

        # Step 5 — compute the target Q-values using the Bellman equation:
        # target = reward + gamma * max(Q(next_state)) * (1 - done)
        # The (1 - done) term zeros out future rewards when the episode ended
        with torch.no_grad():
            next_q   = self.target(next_states).max(dim=1)[0]  # best action in next state
            q_target = rewards + self.gamma * next_q * (1 - dones)
        # shape: (batch,)

        # Step 6 — loss = how wrong were our Q-value predictions?
        loss = nn.MSELoss()(q_predicted, q_target)

        # Step 7 — backpropagation: adjust network weights to reduce the loss
        self.optimizer.zero_grad()   # clear old gradients
        loss.backward()              # compute new gradients
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)  # prevent explosions
        self.optimizer.step()        # update weights

        # Step 8 — decay epsilon (less exploration over time)
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        # Step 9 — periodically copy online → target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
            print(f"  [step {self.step_count}] target network synced | eps={self.eps:.3f}")

        return loss.item()

    def save_model(self, path: str = "dqn_lunarlander.pth"):
        """
        Save the trained network weights to a file.
        Call this after training so you don't have to retrain from scratch.
        """
        checkpoint = {
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.eps,
        }
        torch.save(checkpoint, path)
        print(f"Model saved → {path}")

    def load_model(self, path: str = "dqn_lunarlander.pth"):
        """
        Load previously saved weights from a file.
        Use this to continue training or to run the trained agent.
        """
        if not os.path.exists(path):
            print(f"No saved model found at '{path}' — starting from scratch.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.step_count = checkpoint["step_count"]
        self.eps = checkpoint["epsilon"]
        print(f"Model loaded ← {path}  (step {self.step_count}, eps={self.eps:.3f})")

def train(env, agent: DQNAgent, num_episodes: int = 400):
    """Run the full training loop and return per-episode rewards."""
    all_rewards = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        ep_losses = [] 

        while True:
            # Agent picks an action based on current state
            action = agent.select_action(state)

            # Environment reacts
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # Agent stores experience and learns
            loss = agent.learn(state, action, reward, next_state, done)
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            ep_reward += reward

            if done:
                break

        all_rewards.append(ep_reward)

        avg_loss = np.mean(ep_losses) if ep_losses else 0.0

        # Print progress every 25 episodes
        if ep % 25 == 0:
            avg = np.mean(all_rewards[-25:])  # average of last 25 episodes
            print(f"Episode {ep:4d} | Reward: {ep_reward:8.1f} | "
                  f"Avg(25): {avg:8.1f} | Loss: {avg_loss:.4f} | eps: {agent.eps:.3f}")

    return all_rewards


def test(env, agent: DQNAgent, num_episodes: int = 10):
    """
    Run the agent without any learning or exploration.
    Used to evaluate how good the trained agent actually is.
    """
    print("\n── Testing trained agent ──")
    original_eps = agent.eps
    agent.eps = 0.0   # no randomness during testing — pure exploitation

    rewards = []
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            state = next_state
            ep_reward += reward
            if done:
                break

        rewards.append(ep_reward)
        print(f"  Test episode {ep}: {ep_reward:.1f}")

    print(f"\nMean test reward: {np.mean(rewards):.1f}")
    print("(Score > 200 = solved | Score > 0 = learning | Score < -100 = still random)")

    agent.eps = original_eps  # restore epsilon
    return rewards

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    import gymnasium as gym

    # Verify state → action mapping 
    print("-- Verifying state -> action mapping --")
    env = gym.make("LunarLander-v3")
    test_state, _ = env.reset()

    print(f"  State shape  : {test_state.shape}")   # should be (8,)
    print(f"  State values : {np.round(test_state, 3)}")
    print(f"  Action space : {env.action_space}")    # should be Discrete(4)

    dummy_agent = DQNAgent()
    test_action = dummy_agent.select_action(test_state)
    print(f"  Action chosen: {test_action}  (valid = {0 <= test_action <= 3})")
    print()

    # Train 
    env = gym.make("LunarLander-v3")
    agent = DQNAgent(
        state_dim = 8,
        action_dim = 4,
        lr = 1e-3,
        gamma = 0.99,
        batch_size = 64,
        buffer_size = 50_000,
        target_update_freq = 500,
        eps_start = 1.0,
        eps_end = 0.01,
        eps_decay_steps = 10_000,
        min_buffer = 1_000,
    )

    print("-- Training --")
    training_rewards = train(env, agent, num_episodes=100)
    env.close()

    # Save 
    agent.save_model("dqn_lunarlander.pth")

    # Plot training metrics
    print("\n-- Plotting training metrics --")
    
    def plot_training_rewards(all_rewards):
        plt.figure(figsize=(12, 6))
        plt.plot(all_rewards, alpha=0.6, label="Episode Reward")
        # Moving average
        window = 10
        moving_avg = [np.mean(all_rewards[max(0, i-window):i+1]) for i in range(len(all_rewards))]
        plt.plot(moving_avg, linewidth=2, label=f"Moving Avg ({window} episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress: Reward Per Episode")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    plot_training_rewards(training_rewards)
    
    # Collect final episode to visualize
    print("-- Plotting final episode behavior --")
    from src.env.env_info import STATE_NAMES
    
    def collect_episode_data(agent=None, env_name="LunarLander-v3"):
        env = gym.make(env_name)
        obs, info = env.reset()
        done = False
        history = [[] for _ in range(len(obs))]
        rewards = []
        total_reward = 0.0
        step_count = 0
        while not done:
            for i, val in enumerate(obs):
                history[i].append(val)
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        env.close()
        return history, rewards, total_reward, step_count
    
    def plot_main_states(history):
        plt.figure(figsize=(10, 6))
        plt.plot(history[0], label=STATE_NAMES[0])
        plt.plot(history[1], label=STATE_NAMES[1])
        plt.plot(history[2], label=STATE_NAMES[2])
        plt.plot(history[3], label=STATE_NAMES[3])
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Position and Velocity Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_angle_states(history):
        plt.figure(figsize=(10, 6))
        plt.plot(history[4], label=STATE_NAMES[4])
        plt.plot(history[5], label=STATE_NAMES[5])
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Angle and Angular Velocity Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_leg_contacts(history):
        plt.figure(figsize=(10, 6))
        plt.plot(history[6], label=STATE_NAMES[6])
        plt.plot(history[7], label=STATE_NAMES[7])
        plt.xlabel("Time Step")
        plt.ylabel("Contact")
        plt.title("Leg Contact States Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_reward_curve(rewards):
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Time Step")
        plt.ylabel("Reward Per Step")
        plt.title("Reward Per Step in Final Episode")
        plt.tight_layout()
        plt.show()
    
    history, episode_rewards, total_reward, step_count = collect_episode_data(agent=agent)
    print(f"Final episode - Steps: {step_count}, Total Reward: {total_reward:.2f}")
    plot_main_states(history)
    plot_angle_states(history)
    plot_leg_contacts(history)
    plot_reward_curve(episode_rewards)

    # Test
    env = gym.make("LunarLander-v3", render_mode="human")
    test(env, agent, num_episodes=10)
    env.close()

    print("\n-- Reloading model from disk and retesting --")
    fresh_agent = DQNAgent()
    fresh_agent.load_model("dqn_lunarlander.pth")

    env = gym.make("LunarLander-v3", render_mode="human")
    test(env, fresh_agent, num_episodes=5)
    env.close()