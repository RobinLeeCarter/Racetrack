from typing import Optional, List

import numpy as np

import environment
import policy


class RandomPolicy(policy.Policy):
    # fully random
    def __init__(self, environment_: environment.Environment, rng: np.random.Generator):
        super().__init__(environment_)
        self.rng: np.random.Generator = rng

        # cache state and possible actions for get_probability to avoid doing it twice
        self.state: Optional[environment.State] = None
        self.possible_actions: List[environment.Action] = []

    def get_action_given_state(self, state_: environment.State) -> environment.Action:
        self.set_possible_actions(state_)
        return self.rng.choice(self.possible_actions)

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        self.set_possible_actions(state_)
        return 1.0 / len(self.possible_actions)

    def set_possible_actions(self, state_: environment.State):
        if self.state is None or state_ != self.state:
            # can't use cached version
            self.state = state_
            self.possible_actions = [action for action in self.environment.actions_for_state(state_)]
            if not self.possible_actions:
                raise Exception(f"EGreedyPolicy state: {state_} no possible actions")
