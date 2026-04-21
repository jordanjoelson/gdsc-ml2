import gymnasium as gym
import numpy as np

from src.dqn_agent import DQNAgent
from src.evaluation.plot_rewards import plot_rewards


def train(num_episodes=800):
    """
    Main training loop for DQN agent.

    This function:
    - Creates environment
    - Runs episodes
    - Collects rewards
    - Saves checkpoints at different training stages
    - Returns reward history for visualization
    """

    # Create environment (LunarLander-v3 is the updated version)
    env = gym.make("LunarLander-v3")

    # Initialize agent
    agent = DQNAgent()

    # Store episode rewards for plotting
    rewards = []

    # Loop over episodes
    for ep in range(num_episodes):

        # Reset environment at start of episode
        state, _ = env.reset()

        episode_reward = 0

        while True:
            # Select action using epsilon-greedy policy
            action = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Train agent using experience
            agent.learn(state, action, reward, next_state, done)

            # Move to next state
            state = next_state
            episode_reward += reward

            if done:
                break

        # Store reward
        rewards.append(episode_reward)

        # Print progress every 25 episodes
        if ep % 25 == 0:
            avg = np.mean(rewards[-25:])
            print(f"Episode {ep} | Reward: {episode_reward:.2f} | Avg(25): {avg:.2f}")

        # -------------------------------
        # CHECKPOINT SAVING (FIX ADDED)
        # -------------------------------

        if ep == 100:
            agent.save_model("dqn_early.pth")

        if ep == 400:
            agent.save_model("dqn_mid.pth")

        if ep == num_episodes - 1:
            agent.save_model("dqn_final.pth")

    env.close()

    return rewards


if __name__ == "__main__":
    # Run training
    rewards = train()

    # Plot results
    plot_rewards(rewards)