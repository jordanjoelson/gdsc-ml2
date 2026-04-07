import gymnasium as gym
import matplotlib.pyplot as plt
from src.env.env_info import STATE_NAMES

def collect_episode_data(env_name="LunarLander-v3"):
    env = gym.make(env_name)
    obs, info = env.reset()

    done = False
    history = [[] for _ in range(len(obs))]

    while not done:
        for i, val in enumerate(obs):
            history[i].append(val)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
    return history

def plot_main_states(history):
    plt.figure()
    plt.plot(history[0], label=STATE_NAMES[0])
    plt.plot(history[1], label=STATE_NAMES[1])
    plt.plot(history[2], label=STATE_NAMES[2])
    plt.plot(history[3], label=STATE_NAMES[3])
    plt.legend()
    plt.title("Position & Velocity")
    plt.show()

def plot_angle_states(history):
    plt.figure()
    plt.plot(history[4], label=STATE_NAMES[4])
    plt.plot(history[5], label=STATE_NAMES[5])
    plt.legend()
    plt.title("Angle")
    plt.show()

if __name__ == "__main__":
    history = collect_episode_data()
    plot_main_states(history)
    plot_angle_states(history)