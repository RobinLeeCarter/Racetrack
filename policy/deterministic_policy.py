import numpy as np

import environment
import policy


class DeterministicPolicy(policy.Policy):
    def __init__(self, environment_: environment.Environment, checking_on: bool = True):
        super().__init__(environment_)
        self._action_given_state: np.ndarray = np.empty(shape=self.environment.states_shape, dtype=environment.Action)
        self.checking_on = checking_on

    def set_action(self, state_: environment.State, action_: environment.Action):
        self._action_given_state[state_.index] = action_

    def get_action_given_state(self, state_: environment.State) -> environment.Action:
        action_ = self._action_given_state[state_.index]
        if self.checking_on:
            if self.environment.is_action_compatible_with_state(state_, action_):
                return action_
            else:
                raise Exception(f"DeterministicPolicy state: {state_} not compatible with action: {action_}")
        else:
            return action_

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        deterministic_action = self.get_action_given_state(state_)
        if action_ == deterministic_action:
            return 1.0
        else:
            return 0.0
