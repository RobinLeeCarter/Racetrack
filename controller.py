import numpy as np

import constants

import tracks
import policy
import racetrack
import off_policy_mc_control


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.rng: np.random.Generator = np.random.default_rng()
        self.racetrack_ = racetrack.RaceTrack(tracks.TRACK_1, self.rng)
        self.states_shape = self.racetrack_.track.shape + (constants.MAX_VELOCITY+1, constants.MAX_VELOCITY+1)

        # print(self.states_shape)

        acceleration_range = constants.MAX_ACCELERATION - constants.MIN_ACCELERATION + 1    # 3
        self.actions_shape = (acceleration_range, acceleration_range)   # (3, 3)

        # print(self.actions_shape)

        self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.rng, self.actions_shape)
        self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.states_shape)
        self.algorithm_: off_policy_mc_control.OffPolicyMcControl = off_policy_mc_control.OffPolicyMcControl(
                self.states_shape,
                self.actions_shape,
                self.behaviour_policy,
                self.target_policy
            )

        # self.generate_trajectory()
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
