from typing import List, Optional

import environment
from agent import reward_state_action


class Episode:
    """Just makes a record laid out in the standard way with Reward, State, Action for each t"""
    def __init__(self):
        self.trajectory: List[reward_state_action.RewardStateAction] = []
        self.current: reward_state_action.RewardStateAction = reward_state_action.RewardStateAction(None, None, None)
        self.previous_reward: Optional[float] = None

    def add_normal(self, state: environment.State, action: environment.Action, reward: float):
        # S0, A0, R1, S1, A1, R2 ... S(T-1), A(T-1), R(T)
        rsa = reward_state_action.RewardStateAction(self.previous_reward, state, action)
        self.trajectory.append(rsa)
        self.previous_reward = reward

    def add_terminal(self, state: environment.State):
        rsa = reward_state_action.RewardStateAction(self.previous_reward, state, None)
        self.trajectory.append(rsa)
