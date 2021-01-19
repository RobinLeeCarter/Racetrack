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

    # terminal
    is_terminal: bool = False

    @property
    def index(self) -> tuple:
        return self.x, self.y, self.vx, self.vy
