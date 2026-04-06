import gymnasium as gym
from src.evaluation.plot_rewards import plot_rewards


def run_random_agent(num_episodes=100):
    # Create the LunarLander environment
    env = gym.make("LunarLander-v3")

    # List to store total reward for each episode
    rewards_per_episode = []

    # Loop through each episode
    for episode in range(num_episodes):

        # Reset environment at the start of each episode
        state, _ = env.reset()

        # Track total reward for this episode
        total_reward = 0

        # Flag to check if episode is finished
        done = False

        # Run one full episode
        while not done:

            # Choose a random action (0–3)
            action = env.action_space.sample()

            # Take the action and get results
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Add reward to total
            total_reward += reward

            # Move to next state
            state = next_state

            # Check if episode is finished
            done = terminated or truncated

        # Save total reward for this episode
        rewards_per_episode.append(total_reward)

        # Print result for this episode
        print(f"Episode {episode + 1}: Reward = {total_reward}")

    # Close the environment after training
    env.close()

    # Return all rewards for plotting
    return rewards_per_episode


if __name__ == "__main__":

    # Run training for 100 episodes
    rewards = run_random_agent(100)

    # Plot the rewards to visualize performance
    plot_rewards(rewards)