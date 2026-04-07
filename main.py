import argparse

from demos.demo_env import run_demo
from src.visualization.plot_states import collect_episode_data, plot_main_states, plot_angle_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="demo")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()

    elif args.mode == "plot":
        history = collect_episode_data()
        plot_main_states(history)
        plot_angle_states(history)

    else:
        print("Use --mode demo or --mode plot")

if __name__ == "__main__":
    main()