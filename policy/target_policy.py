import numpy as np

import action
import state
import policy


class TargetPolicy(policy.Policy):
    # deterministic
    def __init__(self, states_shape: tuple):
        self.action_given_state: np.ndarray = np.empty(shape=states_shape, dtype=action.Action)

    def get_action(self, state_: state.State) -> action.Action:
        return self.action_given_state[state_.x, state_.y, state_.vx, state_.vy]
