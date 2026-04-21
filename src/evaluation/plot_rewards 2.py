import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards):
    plt.figure()

    # Raw rewards
    plt.plot(rewards, label="Reward")

    # Moving average (smooth trend)
    window = 10
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg, label="Moving Avg (10)")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.grid(True)

    plt.show()