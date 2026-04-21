import gymnasium as gym
import numpy as np

class RandomAgent:
    """
    Baseline random agent for LunarLander-v2 environment.
    State  (input)  : 8 floats
        [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
    Action (output) : int in {0, 1, 2, 3}
        0 = do nothing
        1 = fire left engine
        2 = fire main engine
        3 = fire right engine
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4):
        self.state_dim  = state_dim   # number of state features
        self.action_dim = action_dim  # number of discrete actions
    # Required interface — every agent you build must implement these

    def select_action(self, state: np.ndarray) -> int:
        """
        Choose an action given the current state.
        Args: state: np.ndarray of shape (8,) — the environment observation
        Returns: action: int in [0, action_dim)
        """
        # Validate the incoming state (helpful for catching env bugs early)
        assert len(state) == self.state_dim, (
            f"Expected state of length {self.state_dim}, got {len(state)}"
        )

        # Random baseline: ignore state, pick uniformly at random
        return np.random.randint(0, self.action_dim)

    def learn(self, state, action, reward, next_state, done):
        """
        Will update DQN logic later
        Args:
            state      : np.ndarray (8,)  — state before the action
            action     : int              — action taken
            reward     : float            — reward received
            next_state : np.ndarray (8,)  — state after the action
            done       : bool             — True if episode ended
        """
        pass  # TODO: store transition in replay buffer, sample & train

# Quick smoke-test (run this file directly to verify the agent works)

if __name__ == "__main__":

    env   = gym.make("LunarLander-v3", render_mode="human")
    agent = RandomAgent(state_dim=8, action_dim=4)

    num_episodes  = 5
    total_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0

        while True:
            action                          = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done                            = term or trunc

            # Call learn() now (no-op) — same call signature you'll use for DQN
            agent.learn(state, action, reward, next_state, done)

            state     = next_state
            ep_reward += reward

            if done:
                break

        total_rewards.append(ep_reward)
        print(f"Episode {ep + 1}: reward = {ep_reward:.1f}")

    print(f"\nMean reward over {num_episodes} episodes: {np.mean(total_rewards):.1f}")
    env.close()