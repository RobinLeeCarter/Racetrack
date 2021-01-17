import numpy as np

import constants

import tracks
import policy
import racetrack
import off_policy_mc_control
from episode import episode


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.rng: np.random.Generator = np.random.default_rng()
        self.racetrack_ = racetrack.RaceTrack(tracks.TRACK_1, self.rng)
        self.states_shape = (self.racetrack_.track.shape[0], self.racetrack_.track.shape[1],
                             constants.MAX_VELOCITY+1, constants.MAX_VELOCITY+1)

        # print(self.states_shape)

        acceleration_range = constants.MAX_ACCELERATION - constants.MIN_ACCELERATION + 1    # 3
        self.actions_shape = (acceleration_range, acceleration_range)   # (3, 3)

        # print(self.actions_shape)

        self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.states_shape)
        self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.rng, self.actions_shape)

        self.algorithm_: off_policy_mc_control.OffPolicyMcControl = off_policy_mc_control.OffPolicyMcControl(
                self.racetrack_,
                self.states_shape,
                self.actions_shape,
                self.behaviour_policy,
                self.target_policy,
                verbose=self.verbose
            )

        # self.generate_trajectory()
        # hyperparameters

        # self.states = np.arange(101)
        # self.non_terminal_states = self.states[1:-1]
        # self.V = np.zeros(shape=self.states.shape, dtype=float)
        # self.policy = np.zeros(shape=self.states.shape, dtype=float)

    def run(self):
        self.algorithm_.run(100_000)
        self.output_q()
        for _ in range(10):
            self.output_example_trajectory()

    def output_q(self):
        q = self.algorithm_.Q
        q_size = q.size
        q_non_zero = np.count_nonzero(q)
        percent_non_zero = 100.0 * q_non_zero / q_size
        print(f"q_size: {q_size}\tq_non_zero: {q_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")

    def output_example_trajectory(self):
        trajectory_ = episode.Episode(self.racetrack_, verbose=True)
        t = 0
        while not trajectory_.is_terminated and not trajectory_.is_grass:
            action_ = self.target_policy.get_action_given_state(trajectory_.current.state)
            print(f"t={t} \t state = {trajectory_.current.state} \t action = {action_}")
            trajectory_.apply_action(action_)
            t += 1
        # trajectory_.output()
        print(f"final position = {trajectory_.current.state.x}, {trajectory_.current.state.y}")
