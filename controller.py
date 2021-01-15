import numpy as np

import constants

import tracks
import policy
import racetrack


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.rng: np.random.Generator = np.random.default_rng()
        self.racetrack_ = racetrack.RaceTrack(tracks.TRACK_1, self.rng)
        self.states_shape = self.racetrack_.track.shape + (constants.MAX_VELOCITY+1, constants.MAX_VELOCITY+1)

        self.behaviour: policy.RandomPolicy = policy.RandomPolicy(self.rng)
        self.target: policy.TargetPolicy = policy.TargetPolicy(self.states_shape)

        self.generate_trajectory()
        # hyperparameters

        # self.states = np.arange(101)
        # self.non_terminal_states = self.states[1:-1]
        # self.V = np.zeros(shape=self.states.shape, dtype=float)
        # self.policy = np.zeros(shape=self.states.shape, dtype=float)

    def run(self):
        i: int = 0
        cont: bool = True

        while cont:
            delta: float = 0.0
            cont = False

    def generate_trajectory(self):
        pass
