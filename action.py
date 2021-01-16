from __future__ import annotations
from dataclasses import dataclass

import constants


@dataclass
class Action:
    # origin at bottom left (max rows)

    # acceleration
    ax: int = 0     # -1 <= ax <= +1
    ay: int = 0     # -1 <= ay <= +1

    @property
    def index(self) -> tuple:
        # ax = ix - constants.MAX_ACCELERATION
        # ay = constants.MAX_ACCELERATION - iy
        ix = self.ax + constants.MAX_ACCELERATION
        iy = constants.MAX_ACCELERATION - self.ay
        return iy, ix

    @staticmethod
    def get_action_from_index(index: tuple) -> Action:
        iy, ix = index
        ax = ix - constants.MAX_ACCELERATION
        ay = constants.MAX_ACCELERATION - iy
        return Action(ax, ay)
