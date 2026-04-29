import gymnasium as gym
import matplotlib.pyplot as plt
from src.env.env_info import STATE_NAMES


# ========================
# DATA COLLECTION
# ========================
def collect_episode_data(env_name="LunarLander-v3", action_fn=None):
    env = gym.make(env_name)
    obs, info = env.reset()

    done = False
    history = [[] for _ in range(len(obs))]
    rewards = []
    total_reward = 0.0

    while not done:
        for i, val in enumerate(obs):
            history[i].append(val)

        # Use provided policy or default to random
        if action_fn is None:
            action = env.action_space.sample()
        else:
            action = action_fn(env, obs)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        total_reward += reward
        done = terminated or truncated

    env.close()
    return history, rewards, total_reward


# ========================
# MOVING AVERAGE (Week 4 IMPORTANT)
# ========================
def moving_average(values, window=20):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i+1]
        result.append(sum(window_vals) / len(window_vals))
    return result


# ========================
# PLOTTING FUNCTIONS
# ========================
def plot_main_states(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history[0], label=STATE_NAMES[0])
    plt.plot(history[1], label=STATE_NAMES[1])
    plt.plot(history[2], label=STATE_NAMES[2])
    plt.plot(history[3], label=STATE_NAMES[3])
    plt.legend()
    plt.title("Position & Velocity")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.show()


def plot_angle_states(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history[4], label=STATE_NAMES[4])
    plt.plot(history[5], label=STATE_NAMES[5])
    plt.legend()
    plt.title("Angle & Angular Velocity")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.show()


def plot_leg_contacts(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history[6], label=STATE_NAMES[6])
    plt.plot(history[7], label=STATE_NAMES[7])
    plt.legend()
    plt.title("Leg Contacts")
    plt.xlabel("Time Step")
    plt.ylabel("Contact")
    plt.show()


def plot_reward_curve(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward per Step", alpha=0.4)
    plt.plot(moving_average(rewards), label="Smoothed Reward")
    plt.legend()
    plt.title("Reward Curve")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.show()


# ========================
# RANDOM POLICY (baseline)
# ========================
def random_policy(env, state):
    return env.action_space.sample()


# ========================
# TEST RUN
# ========================
if __name__ == "__main__":
    history, rewards, total_reward = collect_episode_data()

    print(f"Total Reward: {total_reward:.2f}")

    plot_main_states(history)
    plot_angle_states(history)
    plot_leg_contacts(history)
    plot_reward_curve(rewards)