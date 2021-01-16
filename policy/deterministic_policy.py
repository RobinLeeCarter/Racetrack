import numpy as np

from rsa import action, state
import policy


class DeterministicPolicy(policy.Policy):
    def __init__(self, states_shape: tuple):
        self._action_given_state: np.ndarray = np.empty(shape=states_shape, dtype=action.Action)

    def set_action(self, state_: state.State, action_: action.Action):
        self._action_given_state[state_.y, state_.x, state_.vy, state_.vx] = action_

    def get_action(self, state_: state.State) -> action.Action:
        return self._action_given_state[state_.y, state_.x, state_.vy, state_.vx]

    def get_probability(self, action_: action.Action, state_: state.State) -> float:
        deterministic_action = self.get_action(state_)
        if action_ == deterministic_action:
            return 1.0
        else:
            return 0.0
