from typing import Optional

import numpy as np

import enums
import racetrack
import state


class Trace:
    def __init__(self, racetrack_: racetrack.RaceTrack):
        self.racetrack: racetrack.RaceTrack = racetrack_
        self.trace: Optional[np.ndarray] = None
        self.start()

    def start(self):
        self.trace = self.racetrack.track.copy()

    def mark(self, state_: state.State):
        ix, iy = self.racetrack.get_index(state_.x, state_.y)
        self.trace[iy, ix] = enums.Square.CAR

    def output(self):
        print(self.trace)
