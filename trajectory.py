from typing import Optional, List

import numpy as np

import enums
import racetrack
import action
import state
import state_action_reward


class Trajectory:
    def __init__(self, racetrack_: racetrack.RaceTrack):
        self.racetrack = racetrack_
        self.max_y = self.racetrack.track.shape[0]
        self.max_x = self.racetrack.track.shape[1]

        self.list: List[state_action_reward.StateActionReward] = []
        self.current_state: Optional[state.State] = None

    def add_state(self, state_: state.State):
        self.current_state = state_

    def apply_action(self, action_: action.Action):
        vx = self.current_state.vx + action_.ax
        vy = self.current_state.vy + action_.ay
        x = self.current_state.x + vx
        y = self.current_state.y + vy
        square = self.racetrack.get_square(x, y)
        if square == enums.Square.END:
            # success
            reward = 0.0
        elif square == enums.Square.GRASS:
            reward = -1.0
            x, y = self.racetrack.draw_start_position()
            # failure, move to start line
            pass
        else: # TRACK or START
            # continue
            new_state = state.State(x, y, vx, vy)
            reward = -1.0

    def get_square(self, x: int, y: int) -> enums.Square:
        return self.racetrack.get_square(x, y)


