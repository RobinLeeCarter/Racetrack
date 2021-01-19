from typing import Generator
import numpy as np

import constants
import enums
from environment import action, response, state, track


class Environment:
    def __init__(self, racetrack_: track.RaceTrack, rng: np.random.Generator, verbose: bool = False):
        self.racetrack: track.RaceTrack = racetrack_
        self.rng: np.random.Generator = rng
        self.verbose: bool = verbose

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
        self.trace_ = track.Trace = track.Trace(self.racetrack)

        # pre-reset state (if not None it means the state has just been reset and this was the failure state)
        # self.pre_reset_state: Optional[state.State] = None

    # region Sets
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

    # def current_actions(self) -> Generator[action.Action, None, None]:
    #     yield from self.actions_for_state(self.state)

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
    # endregion

    # region Operation
    def start(self) -> response.Response:
        state_ = self.get_a_start_state()
        if self.verbose:
            self.trace_.start(state_)
        return response.Response(state=state_, reward=0.0)

    def get_a_start_state(self) -> state.State:
        x, y = self.racetrack.get_a_start_position()
        return state.State(x, y)

    def apply_action_to_state(self, state_: state.State, action_: action.Action) -> response.Response:
        if not self.is_action_compatible_with_state(state_, action_):
            raise Exception(f"apply_action_to_state state {state_} incompatible with action {action_}")

        if self.rng.uniform() > constants.SKID_PROBABILITY:
            vx = state_.vx + action_.ax
            vy = state_.vy + action_.ay
        else:  # skid
            vx = state_.vx
            vy = state_.vy
        x = state_.x + vx
        y = state_.y + vy

        square = self.racetrack.get_square(x, y)
        if square == enums.Square.END:
            # success
            reward = 0.0
            state_ = state.State(x, y, vx, vy, is_terminal=True)
            if self.verbose:
                self.trace_.output()
                print(f"Past finish line at {x}, {y}")
        elif square == enums.Square.GRASS:
            # failure, move back to start line
            # self.pre_reset_state = state.State(x, y, vx, vy, is_reset=True)
            reward = -1.0 + constants.EXTRA_REWARD_FOR_FAILURE
            state_ = self.get_a_start_state()
            if self.verbose:
                self.trace_.output()
                print(f"Grass at {x}, {y}")
                self.trace_.start(state_)
        else:
            # TRACK or START so continue
            # self.pre_reset_state = None
            reward = -1.0
            state_ = state.State(x, y, vx, vy)
            if self.verbose:
                self.trace_.mark(state_)

        return response.Response(state_, reward)
    # endregion
