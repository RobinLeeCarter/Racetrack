from dataclasses import dataclass


@dataclass
class Action:
    # origin at bottom left (max rows)

    # acceleration
    ax: int = 0     # -1 <= ax <= +1
    ay: int = 0     # -1 <= ay <= +1
