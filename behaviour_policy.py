import numpy as np

import action


class BehaviourPolicy:
    # fully random
    def __init__(self, rng: np.random.Generator):
        self.rng: np.random.Generator = rng
        self.action_shape = (3, 3)
        self.actions: np.ndarray = np.empty(shape=self.action_shape, dtype=action.Action)
        # self.p: np.ndarray = np.zeros(shape=self.action_shape, dtype=float)
        # self.p[:, :] = 1.0/9.0

        for index, _ in np.ndenumerate(self.actions):
            iy = index[0]
            ix = index[1]
            ax = ix - 1
            ay = 1 - iy
            self.actions[iy, ix] = action.Action(ax, ay)
        self.actions_flattened = self.actions.flatten()

    def draw_action(self) -> action.Action:
        return self.rng.choice(self.actions_flattened)
