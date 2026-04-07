import gymnasium as gym
import time

def run_demo(env_name="LunarLander-v3", episodes=3):
    env = gym.make(env_name, render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        print(f"\nEpisode {ep+1}")

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

            time.sleep(0.02)

        print(f"Total Reward: {total_reward:.2f}")

    env.close()

    if __name__ == "__main__":
        run_demo()