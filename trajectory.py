from typing import List

import enums
import constants

import racetrack
import rsa
import trace
import policy


class Trajectory:
    def __init__(self, racetrack_: racetrack.RaceTrack, verbose: bool = False):
        self.racetrack: racetrack.RaceTrack = racetrack_
        self.verbose: bool = verbose

        self.max_y: int = self.racetrack.track.shape[0]
        self.max_x: int = self.racetrack.track.shape[1]
        self.trace: trace.Trace = trace.Trace(self.racetrack)

        self.episode: List[rsa.RewardStateAction] = []
        self.is_terminated: bool = False
        self.is_grass: bool = False

        self.current: rsa.RewardStateAction = rsa.RewardStateAction(None, None, None)
        self.begin()

    def begin(self):
        # get start position and set state
        x, y = self.racetrack.get_a_start_position()
        self.current.state = rsa.State(x, y)
        if self.verbose:
            self.trace.mark(self.current.state)

    def generate(self, policy_: policy.Policy):
        while not self.is_terminated:
            action_ = policy_.get_action(self.current.state)
            self.apply_action(action_)

    def apply_action(self, action_: rsa.Action):
        # record in list
        self.current.action = action_
        self.episode.append(self.current)

        # apply acceleration to velocity
        vx = self.current.state.vx + action_.ax
        vy = self.current.state.vy + action_.ay
        vx, vy = self.velocity_rules(vx, vy)

        # apply velocity to position
        x = self.current.state.x + vx
        y = self.current.state.y + vy
        square = self.racetrack.get_square(x, y)

        # begin new (reward, state, action)
        self.current = rsa.RewardStateAction(None, None, None)
        if square == enums.Square.END:
            # success
            self.current.reward = 0.0
            self.current.state = rsa.State(x, y, vx, vy, is_terminal=True)
            self.termination()
        elif square == enums.Square.GRASS:
            self.is_grass = True
            # failure, move back to start line
            if self.verbose:
                print(f"Grass at {x}, {y}")
                self.output()
            self.current.reward = -1.0
            x, y = self.racetrack.get_a_start_position()
            vx, vy = 0, 0
            self.current.state = rsa.State(x, y, vx, vy)
            self.trace.start()
            if self.verbose:
                self.trace.mark(self.current.state)
        else:
            # TRACK or START so continue
            self.current.reward = -1.0
            self.current.state = rsa.State(x, y, vx, vy)
            if self.verbose:
                self.trace.mark(self.current.state)

    def termination(self):
        if self.verbose:
            self.output()
        self.episode.append(self.current)
        self.is_terminated = True

    def output(self):
        self.trace.output()

    def velocity_rules(self, vx: int, vy: int) -> tuple:
        vx = self.velocity_bounds(vx)
        vy = self.velocity_bounds(vy)

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

    def velocity_bounds(self, v: int) -> int:
        if v > constants.MAX_VELOCITY:
            return constants.MAX_VELOCITY
        elif v < constants.MIN_VELOCITY:
            return constants.MIN_VELOCITY
        else:
            return v
