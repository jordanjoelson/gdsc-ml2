import gymnasium as gym
import numpy as np

from src.evaluation.plot_rewards import plot_rewards

# Import your existing random agent
from src.random_agent import RandomAgent


def train_agent(num_episodes=300):
    # Create environment
    env = gym.make("LunarLander-v3")

    # Use random agent (for now, until DQN is merged)
    agent = RandomAgent()

    # Store rewards
    rewards_per_episode = []

    for episode in range(num_episodes):
        # Reset environment at start
        state, _ = env.reset()

        total_reward = 0
        done = False

        while not done:
            # Agent picks action (currently random)
            action = agent.select_action(state)

            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Move to next state
            state = next_state

            # Add reward
            total_reward += reward

            # Check if episode is done
            done = terminated or truncated

        # Save reward for this episode
        rewards_per_episode.append(total_reward)

        # Calculate moving average (last 10 episodes)
        avg_reward = np.mean(rewards_per_episode[-10:])

        # Print progress
        print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Avg(10): {avg_reward:.2f}")

        # ---- SAVE CHECKPOINTS (dummy for now) ----
        # Since random agent has no model, we just log progress
        if episode == 50:
            print("Checkpoint: Early stage reached")
        if episode == 150:
            print("Checkpoint: Mid stage reached")
        if episode == 299:
            print("Checkpoint: Final stage reached")

    # Close environment
    env.close()

    return rewards_per_episode


if __name__ == "__main__":
    rewards = train_agent(300)
    plot_rewards(rewards)