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

def print_env_info():
    print("\n=== LunarLander Environment Info ===")

    print("\nState Variables:")
    for i, name in enumerate(STATE_NAMES):
        print(f"{i}: {name}")

    print("\nActions:")
    for k, v in ACTION_MEANINGS.items():
        print(f"{k}: {v}")