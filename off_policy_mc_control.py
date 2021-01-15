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
                 behaviour_policy: policy.RandomPolicy,
                 target_policy: policy.DeterministicPolicy,
                 gamma: float = 1.0
                 ):
        self.racetrack = racetrack_
        self.states_shape = states_shape
        self.actions_shape = actions_shape
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy
        self.gamma = gamma

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
                        self.set_target_policy_to_argmax_q(state_)

    # noinspection PyPep8Naming
    def run(self):
        cont: bool = True
        while cont:
            trajectory_ = trajectory.Trajectory(self.racetrack)
            trajectory_.generate(self.behaviour_policy)
            G: float = 0.0
            W: float = 1.0
            # reversed_non_terminated = reversed(trajectory_.episode[:-1])
            episode = trajectory_.episode
            T: int = len(episode) - 1
            for t in reversed(range(T)):
                R_t_plus_1 = episode[t+1].reward
                S_t = episode[t].state
                A_t = episode[t+1].action
                G = self.gamma * G + R_t_plus_1
                s_a = S_t.tuple + A_t.tuple
                self.C[s_a] += W
                self.Q[s_a] += (W / self.C[s_a]) * (G - self.Q[s_a])
                target_action = self.set_target_policy_to_argmax_q(S_t)
                if A_t.tuple != target_action.tuple:
                    break
                W /= self.behaviour_policy.get_probability(A_t, S_t)

    def set_target_policy_to_argmax_q(self, state_: state.State) -> action.Action:
        """set target_policy to argmax over a of Q breaking ties consistently"""
        # state_index = self.get_index_from_state(state_)
        # print(f"state_index {state_index}")
        state_slice_tuple = state_.tuple + np.s_[:, :]
        q_slice: np.ndarray = self.Q[state_slice_tuple]
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
        return best_action

    # def get_index_from_state(self, state_: state.State) -> tuple:
    #     return state_.x, state_.y, state_.vx, state_.vy

    def get_action_from_index(self, index: tuple) -> action.Action:
        # print(index)
        ix = index[0]
        iy = index[1]
        ax = ix - 1
        ay = iy - 1
        return action.Action(ax, ay)
