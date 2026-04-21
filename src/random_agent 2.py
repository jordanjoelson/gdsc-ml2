import gymnasium as gym
import numpy as np

class RandomAgent:
    """
    Baseline random agent for LunarLander-v2 environment.
    State  (input)  : 8 floats
    Action (output) : int in {0, 1, 2, 3}
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4):
        # --- FIX: handle incorrect input safely ---
        # If someone accidentally passes env.action_space, fix it
        if isinstance(state_dim, gym.spaces.Discrete):
            self.state_dim = 8   # correct state size
            self.action_dim = state_dim.n  # get number of actions
        else:
            self.state_dim = state_dim
            self.action_dim = action_dim

    def select_action(self, state: np.ndarray) -> int:
        """
        Choose an action given the current state.
        """

        # Validate state size
        assert len(state) == self.state_dim, (
            f"Expected state of length {self.state_dim}, got {len(state)}"
        )

        # Random action
        return np.random.randint(0, self.action_dim)

    def learn(self, state, action, reward, next_state, done):
        """
        Will update DQN logic later
        """
        pass