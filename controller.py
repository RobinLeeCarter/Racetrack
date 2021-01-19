import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import enums
import constants
import environment
import policy
import agent
import off_policy_mc_control
import view


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.rng: np.random.Generator = np.random.default_rng()
        self.racetrack = environment.track.RaceTrack(constants.TRACK, self.rng)
        self.environment = environment.Environment(self.racetrack, self.rng, verbose=False)
        self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment, self.rng,
                                                                           greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment, self.rng)
        self.target_agent = agent.Agent(self.environment, self.target_policy)
        self.behaviour_agent = agent.Agent(self.environment, self.behaviour_policy)

        self.algorithm_: off_policy_mc_control.OffPolicyMcControl = off_policy_mc_control.OffPolicyMcControl(
                self.environment,
                self.target_agent,
                self.behaviour_agent,
                verbose=self.verbose
            )
        self.view = view.View(self.racetrack)

    def run(self):
        self.algorithm_.run()
        self.output_q()
        self.graph_samples(self.algorithm_.sample_iteration, self.algorithm_.average_return)

        # self.behaviour_agent.set_policy(self.target_policy)
        self.view.open_window()
        # self.view.display_and_wait()
        # self.environment.verbose = True
        # self.agent.verbose = True

        while True:
            episode_: agent.Episode = self.target_agent.generate_episode()
            user_event: enums.UserEvent = self.view.display_episode(episode_)
            if user_event == enums.UserEvent.QUIT:
                break

    def output_q(self):
        q = self.algorithm_.Q
        q_size = q.size
        q_non_zero = np.count_nonzero(q)
        percent_non_zero = 100.0 * q_non_zero / q_size
        print(f"q_size: {q_size}\tq_non_zero: {q_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")

    def graph_samples(self, iteration: np.ndarray, average_return: np.ndarray):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        ax.set_title("Average Return vs Learning Episodes")
        ax.set_xlim(xmin=0, xmax=constants.LEARNING_EPISODES)
        ax.set_xlabel("Learning Episodes")
        ax.set_ylim(ymin=constants.INITIAL_Q_VALUE*2, ymax=0)
        ax.set_ylabel("Average Return")
        ax.plot(iteration, average_return)
        plt.show()
