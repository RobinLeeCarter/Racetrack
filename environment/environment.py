from typing import Generator

import numpy as np

import racetrack
import constants

from environment import action, state


class Environment:
    def __init__(self, racetrack_: racetrack.RaceTrack):
        self.racetrack: racetrack.RaceTrack = racetrack_

        # position
        self.min_x: int = 0
        self.max_x: int = self.racetrack.track.shape[1] - 1
        self.min_y: int = 0
        self.max_y: int = self.racetrack.track.shape[0] - 1

        # velocity
        self.min_vx: int = 0
        self.max_vx: int = constants.MAX_VELOCITY
        self.min_vy: int = 0
        self.max_vy: int = constants.MAX_VELOCITY

        # acceleration
        self.min_ax: int = constants.MIN_ACCELERATION
        self.max_ax: int = constants.MAX_ACCELERATION
        self.min_ay: int = constants.MIN_ACCELERATION
        self.max_ay: int = constants.MAX_ACCELERATION

        self.state_shape: tuple = (self.max_y + 1, self.max_x + 1, self.max_vy + 1, self.max_vx + 1)
        # self.states: np.ndarray = np.empty(shape=self.state_shape, dtype=state.State)
        # self.build_states()

        self.action_shape: tuple = (self.max_ay - self.min_ay + 1, self.max_ax - self.min_ax + 1)
        # self.actions: np.ndarray = np.empty(shape=self.action_shape, dtype=action.Action)
        # self.build_actions()

    def states(self) -> Generator[state.State, None, None]:
        for y in range(self.state_shape[0]):
            for x in range(self.state_shape[1]):
                for vy in range(self.state_shape[2]):
                    for vx in range(self.state_shape[3]):
                        yield state.State(x, y, vx, vy)

    def actions(self) -> Generator[action.Action, None, None]:
        for iy in range(self.action_shape[0]):
            for ix in range(self.action_shape[1]):
                yield action.Action.get_action_from_index((iy, ix))

    # def build_states(self):
    #     for y in range(self.state_shape[0]):
    #         for x in range(self.state_shape[1]):
    #             for vy in range(self.state_shape[2]):
    #                 for vx in range(self.state_shape[3]):
    #                     self.states[y, x, vy, vx] = state.State(x, y, vx, vy)
    #
    # def build_actions(self):
    #     for iy in range(self.action_shape[0]):
    #         for ix in range(self.action_shape[1]):
    #             self.actions[iy, ix] = action.Action.get_action_from_index((iy, ix))

    def get_a_start_state(self) -> state.State:
        x, y = self.racetrack.get_a_start_position()
        return state.State(x, y)

