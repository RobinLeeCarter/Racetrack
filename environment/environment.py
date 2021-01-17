from typing import Generator, Optional

import numpy as np

import constants
import enums
import racetrack
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

        self.states_shape: tuple = (self.max_x + 1, self.max_y + 1, self.max_vx + 1, self.max_vy + 1)
        self.actions_shape: tuple = (self.max_ax - self.min_ax + 1, self.max_ay - self.min_ay + 1)

        # current state
        self.state: Optional[state.State] = None
        self.reward: float = 0.0

        # pre-reset state (if not None it means the state has just been reset and this was the failure state)
        self.pre_reset_state: Optional[state.State] = None

    def start(self):
        x, y = self.racetrack.get_a_start_position()
        self.state = state.State(x, y)
        self.reward = 0.0
        self.pre_reset_state = None

    def states(self) -> Generator[state.State, None, None]:
        """set S"""
        for x in range(self.states_shape[0]):
            for y in range(self.states_shape[1]):
                for vx in range(self.states_shape[2]):
                    for vy in range(self.states_shape[3]):
                        yield state.State(x, y, vx, vy)

    def actions(self) -> Generator[action.Action, None, None]:
        """set A"""
        for iy in range(self.actions_shape[0]):
            for ix in range(self.actions_shape[1]):
                yield action.Action.get_action_from_index((iy, ix))

    def current_actions(self) -> Generator[action.Action, None, None]:
        yield from self.actions_for_state(self.state)

    # possible need to materialise this if it's slow since it will be at the bottom of the loop
    def actions_for_state(self, state_: state.State) -> Generator[action.Action, None, None]:
        """set A(s)"""
        for action_ in self.actions():
            if self.is_action_compatible_with_state(state_, action_):
                yield action_

    def is_action_compatible_with_state(self, state_: state.State, action_: action.Action):
        new_vx = state_.vx + action_.ax
        new_vy = state_.vy + action_.ay
        if self.min_vx <= new_vx <= self.max_vx and \
            self.min_vy <= new_vy <= self.max_vy and \
                not (new_vx == 0 and new_vy == 0):
            return True
        else:
            return False

    def apply_action(self, action_: action.Action):
        if not self.is_action_compatible_with_state(self.state, action_):
            raise Exception(f"state {self.state} incompatible with action {action_}")

        vx = self.state.vx + action_.ax
        vy = self.state.vy + action_.ay
        x = self.state.x + vx
        y = self.state.y + vy

        square = self.racetrack.get_square(x, y)
        if square == enums.Square.END:
            # success
            self.pre_reset_state = None
            self.reward = 0.0
            self.state = state.State(x, y, vx, vy, is_terminal=True)
        elif square == enums.Square.GRASS:
            # failure, move back to start line
            self.pre_reset_state = state.State(x, y, vx, vy, is_reset=True)
            self.reward = -1.0
            self.state = self.get_a_start_state()
        else:
            # TRACK or START so continue
            self.pre_reset_state = None
            self.reward = -1.0
            self.state = state.State(x, y, vx, vy)
