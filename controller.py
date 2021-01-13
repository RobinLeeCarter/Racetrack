from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import utils
import tracks
import behaviour_policy
import racetrack


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        self.rng: np.random.Generator = np.random.default_rng()
        self.behaviour: behaviour_policy.BehaviourPolicy = behaviour_policy.BehaviourPolicy(self.rng)
        self.racetrack: racetrack.RaceTrack(tracks.TRACK_1, self.rng)

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

