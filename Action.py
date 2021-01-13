from dataclasses import dataclass


@dataclass
class Action:
    # origin at bottom left (max rows)
    ax: int = 0
    ay: int = 0
