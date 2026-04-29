import argparse

from demos.demo_env import run_demo
from src.visualization.plot_states import (
    collect_episode_data,
    plot_main_states,
    plot_angle_states,
    plot_leg_contacts,
    plot_reward_curve,
)
from src.env.env_info import print_env_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="demo")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()

    elif args.mode == "plot":
        history, rewards, total_reward = collect_episode_data()
        print(f"Total Reward: {total_reward:.2f}")

        plot_main_states(history)
        plot_angle_states(history)
        plot_leg_contacts(history)
        plot_reward_curve(rewards)

    elif args.mode == "info":
        print_env_info()

    else:
        print("Use --mode demo, --mode plot, or --mode info")


if __name__ == "__main__":
    main()