from typing import Optional

import numpy as np

import racetrack
import policy
import state
import action
import trajectory


class OffPolicyMcControl:
    def __init__(self,
                 racetrack_: racetrack.RaceTrack,
                 states_shape: tuple,
                 actions_shape: tuple,
                 behaviour_policy: policy.Policy,
                 target_policy: policy.DeterministicPolicy,
                 gamma: float = 1.0,
                 verbose: bool = False
                 ):
        self.racetrack: racetrack.RaceTrack = racetrack_
        self.states_shape: tuple = states_shape
        self.actions_shape: tuple = actions_shape
        self.behaviour_policy: policy.Policy = behaviour_policy
        self.target_policy: policy.DeterministicPolicy = target_policy
        self.gamma = gamma
        self.verbose = verbose

        self.center_action_flat_index: int = self.find_center_action_flat_index()
        self.Q: np.ndarray = np.zeros(shape=states_shape + actions_shape, dtype=float)
        self.Q.fill(-100)  # so that a successful trajectory is always better
        self.C: np.ndarray = np.zeros(shape=self.Q.shape, dtype=float)
        self.initialise_target_policy()

    def initialise_target_policy(self):
        for y in range(self.states_shape[0]):
            if self.verbose:
                print(f"y = {y}")
            for x in range(self.states_shape[1]):
                for vy in range(self.states_shape[2]):
                    for vx in range(self.states_shape[3]):
                        state_ = state.State(x, y, vx, vy)
                        self.set_target_policy_to_argmax_q(state_)
                        print(f"s={state_} -> a={self.target_policy.get_action(state_)}")

    def find_center_action_flat_index(self) -> Optional[int]:
        for yi in range(self.actions_shape[0]):
            for xi in range(self.actions_shape[1]):
                action_ = action.Action.get_action_from_index((yi, xi))
                if action_.ax == 0 and action_.ay == 0:
                    return yi * self.actions_shape[1] + xi
        raise Exception("center_action_flat_index not found")

    # noinspection PyPep8Naming
    def run(self, iterations: int):
        i: int = 0
        while i <= iterations:
            if self.verbose:
                print(f"iteration = {i}")
            else:
                if i % 10000 == 0:
                    print(f"iteration = {i}")
            trajectory_ = trajectory.Trajectory(self.racetrack)
            trajectory_.generate(self.behaviour_policy)
            G: float = 0.0
            W: float = 1.0
            # reversed_non_terminated = reversed(trajectory_.episode[:-1])
            episode = trajectory_.episode
            T: int = len(episode) - 1
            # print(f"T = {T}")
            for t in reversed(range(T)):
                # if t < T-1:
                #     print(f"t = {t}")
                R_t_plus_1 = episode[t+1].reward
                S_t = episode[t].state
                A_t = episode[t].action
                G = self.gamma * G + R_t_plus_1
                s_a = S_t.index + A_t.index
                # print(f"s_a = {s_a}")
                self.C[s_a] += W
                self.Q[s_a] += (W / self.C[s_a]) * (G - self.Q[s_a])
                target_action = self.set_target_policy_to_argmax_q(S_t)
                print(f"S_t={S_t} -> new_a={target_action}")
                if A_t.index != target_action.index:
                    break
                W /= self.behaviour_policy.get_probability(A_t, S_t)
            i += 1

    def set_target_policy_to_argmax_q(self, state_: state.State) -> action.Action:
        """set target_policy to argmax over a of Q breaking ties consistently"""
        # state_index = self.get_index_from_state(state_)
        # print(f"state_index {state_index}")
        state_slice_tuple = state_.index + np.s_[:, :]
        q_slice: np.ndarray = self.Q[state_slice_tuple]
        # print(f"q_slice.shape {q_slice.shape}")

        # argmax
        best_q = np.max(q_slice)
        # print(f"best_q {best_q}")
        best_q_bool = (q_slice == best_q)
        # print(f"best_q_bool.shape {best_q_bool.shape}")
        best_flat_indexes = np.flatnonzero(best_q_bool)

        if self.center_action_flat_index in best_flat_indexes:
            consistent_best_flat_index = self.center_action_flat_index  # ax = 0, ay = 0
        else:
            consistent_best_flat_index = np.flatnonzero(best_q_bool)[0]

        # print(f"consistent_best_flat_index {consistent_best_flat_index}")
        best_index = np.unravel_index(consistent_best_flat_index, shape=q_slice.shape)
        # print(f"best_index {best_index}")
        best_action = action.Action.get_action_from_index(best_index)
        # best_action = self.get_action_from_index(best_index)
        # print(f"best_action {best_action}")
        self.target_policy.set_action(state_, best_action)
        return best_action

    # def get_index_from_state(self, state_: state.State) -> index:
    #     return state_.x, state_.y, state_.vx, state_.vy

    # def get_action_from_index(self, index: tuple) -> action.Action:
    #     # print(index)
    #     iy = index[0]
    #     ix = index[1]
    #     ax = ix - 1
    #     ay = iy - 1
    #     return action.Action(ax, ay)
