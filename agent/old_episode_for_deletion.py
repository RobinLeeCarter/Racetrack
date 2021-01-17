from typing import List

import enums

import environment
from environment import trace
import policy
from agent import reward_state_action


class Episode:
    def __init__(self, environment_: environment.Environment, policy_: policy.Policy, verbose: bool = False):
        self.environment: environment.Environment = environment_
        self.policy: policy.Policy = policy_
        self.verbose: bool = verbose

        self.trace: trace.Trace = trace.Trace(self.environment.racetrack)

        self.trajectory: List[reward_state_action.RewardStateAction] = []
        self.is_terminated: bool = False
        self.is_grass: bool = False

        self.current: reward_state_action.RewardStateAction = reward_state_action.RewardStateAction(None, None, None)

    def generate(self):
        self.environment.start()
        if self.verbose:
            self.trace.mark(self.environment.state)

        while not self.is_terminated:
            # get action
            current_actions = self.environment.current_actions()
            action_ = self.policy.get_action_given_state(self.environment.state, current_actions)

            # record in list
            self.current.action = action_
            self.trajectory.append(self.current)

            # apply action
            self.environment.apply_action(action_)

            if self.verbose:
                pass

    def apply_action(self, action_: environment.Action):
        # record in list
        self.current.action = action_
        self.trajectory.append(self.current)

        # apply acceleration to velocity
        vx = self.current.state.vx + action_.ax
        vy = self.current.state.vy + action_.ay
        vx, vy = self.velocity_rules(vx, vy)

        # apply velocity to position
        x = self.current.state.x + vx
        y = self.current.state.y + vy
        square = self.racetrack.get_square(x, y)

        # begin new (reward, state, action)
        self.current = environment.RewardStateAction(None, None, None)
        if square == enums.Square.END:
            # success
            self.current.reward = 0.0
            self.current.state = environment.State(x, y, vx, vy, is_terminal=True)
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
            self.current.state = environment.State(x, y, vx, vy)
            self.trace.start()
            if self.verbose:
                self.trace.mark(self.current.state)
        else:
            # TRACK or START so continue
            self.current.reward = -1.0
            self.current.state = environment.State(x, y, vx, vy)
            if self.verbose:
                self.trace.mark(self.current.state)

    def termination(self):
        if self.verbose:
            self.output()
        self.trajectory.append(self.current)
        self.is_terminated = True

    def output(self):
        self.trace.output()
