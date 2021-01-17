import numpy as np

import tracks
import racetrack
import environment
import policy
import agent
import off_policy_mc_control


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.rng: np.random.Generator = np.random.default_rng()
        self.racetrack = racetrack.RaceTrack(tracks.TRACK_1, self.rng)
        self.environment = environment.Environment(self.racetrack, verbose=False)
        self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        # self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment, self.rng,
        #                                                                    greedy_policy=self.target_policy)
        self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment, self.rng)
        self.agent = agent.Agent(self.environment, self.behaviour_policy)

        self.algorithm_: off_policy_mc_control.OffPolicyMcControl = off_policy_mc_control.OffPolicyMcControl(
                self.environment,
                self.agent,
                self.target_policy,
                verbose=self.verbose
            )

    def run(self):
        self.algorithm_.run(1_000_00)
        self.output_q()

        self.agent.set_policy(self.target_policy)
        # self.target_policy.checking_on = True
        self.environment.verbose = True
        for _ in range(10):
            print()
            self.agent.generate_episode()

    def output_q(self):
        q = self.algorithm_.Q
        q_size = q.size
        q_non_zero = np.count_nonzero(q)
        percent_non_zero = 100.0 * q_non_zero / q_size
        print(f"q_size: {q_size}\tq_non_zero: {q_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")

    # def output_example_trajectory(self):
    #     episode = self.target_agent
    #     trajectory_ = episode.Episode(self.racetrack_, verbose=True)
    #     t = 0
    #     while not trajectory_.is_terminated and not trajectory_.is_grass:
    #         action_ = self.target_policy.get_action_given_state(trajectory_.current.state)
    #         print(f"t={t} \t state = {trajectory_.current.state} \t action = {action_}")
    #         trajectory_.apply_action(action_)
    #         t += 1
    #     # trajectory_.output()
    #     print(f"final position = {trajectory_.current.state.x}, {trajectory_.current.state.y}")
