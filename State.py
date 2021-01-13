from dataclasses import dataclass


@dataclass
class State:
    # origin at bottom left (max rows)
    x: int
    y: int
    vx: int = 0
    vy: int = 0
