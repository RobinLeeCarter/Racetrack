from typing import Optional

import numpy as np

import enums
from environment.track import racetrack
import environment


class Trace:
    def __init__(self, racetrack_: racetrack.RaceTrack):
        self.racetrack: racetrack.RaceTrack = racetrack_
        self.trace: Optional[np.ndarray] = None

    def start(self, state_: environment.State):
        self.trace = self.racetrack.track.copy()
        self.mark(state_)

    def mark(self, state_: environment.State):
        iy, ix = self.racetrack.get_index(state_.x, state_.y)
        self.trace[iy, ix] = enums.Square.CAR

    def output(self):
        print(self.trace)
