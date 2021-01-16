import numpy as np

from rsa import action, state
import policy


class EGreedyPolicy(policy.Policy):
    # e-greedy
    def __init__(self, rng: np.random.Generator, actions_shape: tuple,
                 greedy_policy: policy.Policy, epsilon: float = 0.1):
        self.rng: np.random.Generator = rng
        self.greedy_policy: policy.Policy = greedy_policy
        self.epsilon = epsilon

        self.actions: np.ndarray = np.empty(shape=actions_shape, dtype=action.Action)
        self.non_greedy_p = self.epsilon * (1.0 / self.actions.size)
        self.greedy_p = (1 - self.epsilon) + self.non_greedy_p

        for index, _ in np.ndenumerate(self.actions):
            self.actions[index] = action.Action.get_action_from_index(index)
        self.actions_flattened = self.actions.flatten()

    def get_action(self, state_: state.State) -> action.Action:
        if self.rng.uniform() > self.epsilon:
            return self.greedy_policy.get_action(state_)
        else:
            return self.rng.choice(self.actions_flattened)

    def get_probability(self, action_: action.Action, state_: state.State) -> float:
        greedy_action = self.greedy_policy.get_action(state_)
        if action_ == greedy_action:
            return self.greedy_p
        else:
            return self.non_greedy_p
