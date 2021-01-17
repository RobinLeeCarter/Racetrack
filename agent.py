import environment
import policy
import episode


class Agent:
    def __init__(self, environment_: environment.Environment, policy_: policy.Policy):
        self.environment = environment_
        self.policy = policy_

    def set_policy(self, policy_: policy.Policy):
        self.policy = policy_

    def generate_episode(self):
        episode_ = episode.Episode(self.environment, self.policy)
        episode_.generate()
