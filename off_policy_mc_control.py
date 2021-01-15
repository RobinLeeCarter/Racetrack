import numpy as np

import policy
import state
import action


class OffPolicyMcControl:
    def __init__(self,
                 states_shape: tuple,
                 actions_shape: tuple,
                 behaviour_policy: policy.RandomPolicy,
                 target_policy: policy.DeterministicPolicy
                 ):
        self.states_shape = states_shape
        self.actions_shape = actions_shape
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy

        self.Q: np.ndarray = np.zeros(shape=states_shape + actions_shape, dtype=float)
        self.C: np.ndarray = np.zeros(shape=self.Q.shape, dtype=float)
        self.initialise_target_policy()

    def initialise_target_policy(self):
        for x in range(self.states_shape[0]):
            # print(f"x = {x}")
            for y in range(self.states_shape[1]):
                for vx in range(self.states_shape[2]):
                    for vy in range(self.states_shape[3]):
                        state_ = state.State(x, y, vx, vy)
                        self.set_target_policy(state_)

    def run(self):
        pass

    def set_target_policy(self, state_: state.State):
        """set target_policy to argmax over a of Q breaking ties consistently"""
        state_index = self.get_index_from_state(state_)
        # print(f"state_index {state_index}")
        slice_tuple = state_index + np.s_[:, :]
        q_slice: np.ndarray = self.Q[slice_tuple]
        # print(f"q_slice.shape {q_slice.shape}")

        # argmax
        best_q = np.max(q_slice)
        # print(f"best_q {best_q}")
        best_q_bool = (q_slice == best_q)
        # print(f"best_q_bool.shape {best_q_bool.shape}")
        consistent_best_flat_index = np.flatnonzero(best_q_bool)[0]
        # print(f"consistent_best_flat_index {consistent_best_flat_index}")
        best_index = np.unravel_index(consistent_best_flat_index, shape=q_slice.shape)
        # print(f"best_index {best_index}")
        best_action = self.get_action_from_index(best_index)
        # print(f"best_action {best_action}")
        self.target_policy.set_action(state_, best_action)

    def get_index_from_state(self, state_: state.State) -> tuple:
        return state_.x, state_.y, state_.vx, state_.vy

    def get_action_from_index(self, index: tuple) -> action.Action:
        # print(index)
        ix = index[0]
        iy = index[1]
        ax = ix - 1
        ay = iy - 1
        return action.Action(ax, ay)
