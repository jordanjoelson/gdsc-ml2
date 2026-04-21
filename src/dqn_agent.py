import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque


# -----------------------------
# Neural Network (Q-function)
# -----------------------------
class QNetwork(nn.Module):
    """
    Simple feedforward neural network that maps:
    state (8 values) -> Q-values for 4 actions
    """

    def __init__(self, state_dim=8, action_dim=4, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    """
    Stores past experiences so the agent can learn from them later.
    """

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Save one transition
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample experiences for training stability
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# DQN Agent
# -----------------------------
class DQNAgent:
    """
    Deep Q-Network agent:
    - chooses actions
    - learns from replay buffer
    - updates neural network
    """

    def __init__(self):

        # Basic hyperparameters
        self.state_dim = 8
        self.action_dim = 4
        self.gamma = 0.99

        self.batch_size = 64
        self.min_buffer = 1000

        # Epsilon-greedy settings
        self.eps = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.0001

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)

        self.target.load_state_dict(self.online.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.online.parameters(), lr=1e-3)

        # Replay memory
        self.buffer = ReplayBuffer()

        # Step counter
        self.step_count = 0

    # -----------------------------
    # Action selection
    # -----------------------------
    def select_action(self, state):

        # Explore (random action)
        if random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)

        # Exploit (best action from Q-network)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.online(state_tensor)

        return torch.argmax(q_values).item()

    # -----------------------------
    # Learning step
    # -----------------------------
    def learn(self, state, action, reward, next_state, done):

        # Store experience
        self.buffer.push(state, action, reward, next_state, done)

        # Wait until buffer is large enough
        if len(self.buffer) < self.min_buffer:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q = self.target(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        # Sync target network occasionally
        self.step_count += 1
        if self.step_count % 500 == 0:
            self.target.load_state_dict(self.online.state_dict())

    # -----------------------------
    # Save model
    # -----------------------------
    def save_model(self, path):
        torch.save(self.online.state_dict(), path)
        print(f"Saved model -> {path}")

    # -----------------------------
    # Load model
    # -----------------------------
    def load_model(self, path):
        if os.path.exists(path):
            self.online.load_state_dict(torch.load(path))
            self.target.load_state_dict(self.online.state_dict())
            print(f"Loaded model <- {path}")