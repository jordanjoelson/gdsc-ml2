import gymnasium as gym
import matplotlib.pyplot as plt
from src.env.env_info import STATE_NAMES


def collect_episode_data(env_name="LunarLander-v3"):
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

        action = env.action_space.sample()  # random baseline
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
    plt.ylabel("Reward")
    plt.title("Reward Per Step")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    history, rewards, total_reward, step_count = collect_episode_data()
    print(f"Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")

    plot_main_states(history)
    plot_angle_states(history)
    plot_leg_contacts(history)
    plot_reward_curve(rewards)