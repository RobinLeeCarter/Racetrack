from typing import Optional, List

import numpy as np

import enums
import racetrack
import action
import state
import trace
import reward_state_action


class Trajectory:
    def __init__(self, racetrack_: racetrack.RaceTrack):
        self.racetrack: racetrack.RaceTrack = racetrack_
        self.max_y: int = self.racetrack.track.shape[0]
        self.max_x: int = self.racetrack.track.shape[1]
        self.trace: trace.Trace = trace.Trace(self.racetrack)

        self.list: List[reward_state_action.RewardStateAction] = []
        self.is_terminated: bool = False

        self.current: reward_state_action.RewardStateAction = reward_state_action.RewardStateAction(None, None, None)
        self.begin()

    def begin(self):
        # get start position and set state
        x, y = self.racetrack.get_a_start_position()
        self.current.state = state.State(x, y)
        self.trace.mark(self.current.state)

    def apply_action(self, action_: action.Action):
        # record in list
        self.current.action = action_
        self.list.append(self.current)

        # apply acceleration to velocity
        vx = self.current.state.vx + action_.ax
        vy = self.current.state.vy + action_.ay
        vx, vy = self.velocity_rules(vx, vy)

        # apply velocity to position
        x = self.current.state.x + vx
        y = self.current.state.y + vy
        square = self.racetrack.get_square(x, y)

        # begin new (reward, state, action)
        self.current = reward_state_action.RewardStateAction(None, None, None)
        if square == enums.Square.END:
            # success
            self.current.reward = 0.0
            self.current.state = state.State(0, 0, 0, 0, is_terminal=True)
            self.termination()
        elif square == enums.Square.GRASS:
            # failure, move back to start line
            print(f"Grass at {x}, {y}")
            self.trace.output()

            self.current.reward = -1.0
            x, y = self.racetrack.get_a_start_position()
            vx, vy = 0, 0
            self.current.state = state.State(x, y, vx, vy)
            self.trace.start()
            self.trace.mark(self.current.state)
        else:
            # TRACK or START so continue
            self.current.reward = -1.0
            self.current.state = state.State(x, y, vx, vy)
            self.trace.mark(self.current.state)

    def termination(self):
        self.trace.output()
        self.list.append(self.current)
        self.is_terminated = True

    def velocity_rules(self, vx: int, vy: int) -> tuple:
        vx = self.max_speed(vx)
        vy = self.max_speed(vy)

        if vx == 0 and vy == 0:
            prev_vx = self.current.state.vx
            prev_vy = self.current.state.vy
            if prev_vx == 0:
                if prev_vy == 0:
                    vx = 0
                    vy = 1
                else:
                    vx = 0
                    vy = prev_vy
            else:
                vx = prev_vx
                vy = 0

        assert not (vx == 0 and vy == 0)

        return vx, vy

    def max_speed(self, v: int) -> int:
        if v > 5:
            return 5
        elif v < -5:
            return -5
        else:
            return v
