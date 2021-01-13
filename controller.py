from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import utils
import constants
import racetracks


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        # hyperparameters
        # breaking ties
        self.action_value_round_dp: int = 7
        # accuracy
        self.theta_power_of_ten: int = -7
        self.theta = 10**self.theta_power_of_ten
        # self.p_heads = 0.5   # probability heads original
        # self.p_heads = 0.25  # probability heads first variation
        self.p_heads = 0.55  # probability heads second variation
        self.gamma = 1.0     # 0.99... is interesting
        self.minimal_action: bool = False    # take the minimal wager of the argmax or the maximal wager

        self.states = np.arange(101)
        self.non_terminal_states = self.states[1:-1]
        self.V = np.zeros(shape=self.states.shape, dtype=float)
        self.policy = np.zeros(shape=self.states.shape, dtype=float)

    def run(self):
        i: int = 0
        cont: bool = True

        while cont:
            delta: float = 0.0
            cont = False
