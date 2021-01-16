import numpy as np

import constants
import racetrack
from environment import state


class States:
    def __init__(self, racetrack_: racetrack.RaceTrack):
        self.racetrack: racetrack.RaceTrack = racetrack_
        self.max_y: int = self.racetrack.track.shape[0] - 1
        self.max_x: int = self.racetrack.track.shape[1] - 1
        self.max_vy: int = constants.MAX_VELOCITY
        self.max_vx: int = constants.MAX_VELOCITY

        self.state_shape: tuple = (self.max_y + 1, self.max_x + 1, self.max_vy + 1, self.max_vx + 1)
        self.states: np.ndarray = np.empty(shape=self.state_shape, dtype=state.State)
        self.build()

    def build(self):
        for y in range(self.state_shape[0]):
            for x in range(self.state_shape[1]):
                for vy in range(self.state_shape[2]):
                    for vx in range(self.state_shape[3]):
                        self.states[y, x, vy, vx] = state.State(x, y, vx, vy)
