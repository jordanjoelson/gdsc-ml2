import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards):
    # Create a new figure for the graph
    plt.figure()

    # Plot raw rewards (this is noisy but shows exact performance)
    plt.plot(rewards, label="Reward per Episode")

    # ---- Moving Average (smooth trend line) ----
    # This helps us see the general trend more clearly
    window_size = 10

    # Compute moving average over rewards
    moving_avg = np.convolve(
        rewards,
        np.ones(window_size) / window_size,
        mode="valid"
    )

    # Plot the smoothed reward trend
    plt.plot(moving_avg, label="Moving Average (Based on last 10 episodes)")

    # Label x-axis
    plt.xlabel("Episode")

    # Label y-axis
    plt.ylabel("Total Reward")

    # Title of the graph
    plt.title("Reward per Episode")

    # Show legend so we know which line is which
    plt.legend()

    # Add grid for easier reading
    plt.grid(True)

    # Display the plot
    plt.show()