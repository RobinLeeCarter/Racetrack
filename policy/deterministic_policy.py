import numpy as np

import action
import state
import policy


class DeterministicPolicy(policy.Policy):
    def __init__(self, states_shape: tuple):
        self._action_given_state: np.ndarray = np.empty(shape=states_shape, dtype=action.Action)

    def set_action(self, state_: state.State, action_: action.Action):
        self._action_given_state[state_.x, state_.y, state_.vx, state_.vy] = action_

    def get_action(self, state_: state.State) -> action.Action:
        return self._action_given_state[state_.x, state_.y, state_.vx, state_.vy]
