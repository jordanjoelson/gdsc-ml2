STATE_NAMES = [
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact"
]

ACTION_MEANINGS = {
    0: "do nothing",
    1: "fire left orientation engine",
    2: "fire main engine",
    3: "fire right orientation engine"
}

REWARD_DESCRIPTION = [
    "Reward increases as the lander gets closer to the landing pad.",
    "Reward increases for reducing velocity (slower movement is better).",
    "Reward increases for keeping the lander upright (low angle).",
    "Successful landing gives a large positive reward.",
    "Crashing results in a large negative reward.",
    "Using engines consumes fuel and may reduce reward slightly."
]


def print_env_info():
    print("\n=== LunarLander Environment Info ===")

    print("\nState Variables (8 values):")
    for i, name in enumerate(STATE_NAMES):
        print(f"{i}: {name}")

    print("\nActions (0–3):")
    for action, meaning in ACTION_MEANINGS.items():
        print(f"{action}: {meaning}")

    print("\nReward Behavior:")
    for line in REWARD_DESCRIPTION:
        print(f"- {line}")