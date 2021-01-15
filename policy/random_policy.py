import numpy as np

import action
import state
import policy
import constants


class RandomPolicy(policy.Policy):
    # fully random
    def __init__(self, rng: np.random.Generator, actions_shape: tuple):
        self.rng: np.random.Generator = rng
        self.actions: np.ndarray = np.empty(shape=actions_shape, dtype=action.Action)
        # self.p: np.ndarray = np.zeros(shape=self.action_shape, dtype=float)
        # self.p[:, :] = 1.0/9.0
        self.p = 1.0/self.actions.size

        for index, _ in np.ndenumerate(self.actions):
            iy = index[0]
            ix = index[1]
            ax = ix - constants.MAX_ACCELERATION
            ay = constants.MAX_ACCELERATION - iy
            self.actions[iy, ix] = action.Action(ax, ay)
        self.actions_flattened = self.actions.flatten()

    def get_action(self, state_: state.State) -> action.Action:
        return self.rng.choice(self.actions_flattened)

    def get_probability(self, action_: action.Action, state_: state.State) -> float:
        return self.p
