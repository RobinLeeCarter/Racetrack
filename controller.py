import numpy as np

import enums
import environment
import policy
import agent
import off_policy_mc_control
import view


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.rng: np.random.Generator = np.random.default_rng()
        self.racetrack = environment.track.RaceTrack(environment.track.TRACK_2, self.rng)
        self.environment = environment.Environment(self.racetrack, self.rng, verbose=False)
        self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment, self.rng,
                                                                           greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment, self.rng)
        self.agent = agent.Agent(self.environment, self.behaviour_policy)

        self.algorithm_: off_policy_mc_control.OffPolicyMcControl = off_policy_mc_control.OffPolicyMcControl(
                self.environment,
                self.agent,
                self.target_policy,
                verbose=self.verbose
            )
        self.view = view.View(self.racetrack)

    def run(self):
        self.algorithm_.run(100_000)
        self.output_q()

        self.agent.set_policy(self.target_policy)
        self.view.open_window()
        # self.view.display_and_wait()
        # self.environment.verbose = True
        # self.agent.verbose = True
        for _ in range(10):
            # print()
            episode_: agent.Episode = self.agent.generate_episode()
            user_event: enums.UserEvent = self.view.display_episode(episode_)
            if user_event == enums.UserEvent.QUIT:
                break

    def output_q(self):
        q = self.algorithm_.Q
        q_size = q.size
        q_non_zero = np.count_nonzero(q)
        percent_non_zero = 100.0 * q_non_zero / q_size
        print(f"q_size: {q_size}\tq_non_zero: {q_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")
