from __future__ import annotations
from dataclasses import dataclass

import constants


@dataclass
class Action:
    # acceleration
    ax: int = 0     # -1 <= ax <= +1
    ay: int = 0     # -1 <= ay <= +1

    @property
    def index(self) -> tuple:
        # ax = ix - constants.MAX_ACCELERATION
        # ay = constants.MAX_ACCELERATION - iy
        ix = self.ax - constants.MIN_ACCELERATION
        iy = self.ay - constants.MIN_ACCELERATION
        return ix, iy

    @staticmethod
    def get_action_from_index(index: tuple) -> Action:
        ix, iy = index
        ax = ix + constants.MIN_ACCELERATION
        ay = iy + constants.MIN_ACCELERATION
        return Action(ax, ay)
