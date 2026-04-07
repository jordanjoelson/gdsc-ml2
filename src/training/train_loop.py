import os
import pickle
import numpy as np
import gymnasium as gym
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.training.random_agent import RandomAgent
from src.evaluation.plot_rewards import plot_rewards

def run_agent_training(num_episodes=100, checkpoint_every=25):
    # Create the LunarLander environment
    env = gym.make("LunarLander-v3")

    # ── Task: Plug the agent into the training loop ──
    agent = RandomAgent(state_dim=8, action_dim=4)

    # List to store total reward for each episode
    rewards_per_episode = []

    # Create folder to save checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    # ── Task: Run multiple episodes for training ──
    for episode in range(num_episodes):

        # Reset environment at the start of each episode
        state, _ = env.reset()

        # Track total reward for this episode
        total_reward = 0

        # Flag to check if episode is finished
        done = False

        # Run one full episode
        while not done:

            # ── Agent picks action (not random anymore) ──
            action = agent.select_action(state)

            # Take the action and get results
            next_state, reward, terminated, truncated, _ = env.step(action)

            # ── Agent learns from this step ──
            agent.learn(state, action, reward, next_state, terminated or truncated)

            # Add reward to total
            total_reward += reward

            # Move to next state
            state = next_state

            # Check if episode is finished
            done = terminated or truncated

        # Save total reward for this episode
        rewards_per_episode.append(total_reward)

        # ── Task: Log total and average rewards per episode ──
        avg_reward = np.mean(rewards_per_episode)
        print(f"Episode {episode + 1}: Reward = {total_reward:.1f} | Avg so far = {avg_reward:.1f}")

        # ── Task: Save checkpoints every N episodes ──
        if (episode + 1) % checkpoint_every == 0:
            checkpoint_path = f"checkpoints/agent_ep{episode + 1}.pkl"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(agent, f)
            print(f"  ✓ Checkpoint saved → {checkpoint_path}")

    # Close the environment
    env.close()

    # Final summary
    print(f"\nTraining complete!")
    print(f"Total episodes : {num_episodes}")
    print(f"Average reward : {np.mean(rewards_per_episode):.1f}")
    print(f"Best episode   : {max(rewards_per_episode):.1f}")

    return rewards_per_episode


if __name__ == "__main__":

    # Run training for 100 episodes, save checkpoint every 25
    rewards = run_agent_training(num_episodes=100, checkpoint_every=25)

    # Plot the rewards to visualize performance
    plot_rewards(rewards)