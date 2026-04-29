import time
import gymnasium as gym


def run_demo(env_name="LunarLander-v3", episodes=3, delay=0.02):
    env = gym.make(env_name, render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        print(f"\nEpisode {ep + 1}")

        while not done:
            action = env.action_space.sample()  # random baseline action
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1
            done = terminated or truncated

            time.sleep(delay)

        print(f"Steps: {step_count}")
        print(f"Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    run_demo()