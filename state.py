from dataclasses import dataclass


@dataclass
class State:
    # origin at bottom left (max rows)

    # position
    x: int          # >= 0
    y: int          # >= 0

    # velocity
    # NOT vx == 0 AND vy ==0 except at start
    vx: int = 0     # 0 <= vx <= 5
    vy: int = 0     # 0 <= vy <= 5
