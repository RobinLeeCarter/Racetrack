from typing import Optional

import environment
import policy
from agent import episode


class Agent:
    def __init__(self, environment_: environment.Environment, policy_: policy.Policy, verbose: bool = False):
        self.environment: environment.Environment = environment_
        self.policy: policy.Policy = policy_
        self.verbose: bool = verbose

        self.state: Optional[environment.State] = None
        self.action: Optional[environment.Action] = None
        self.response: Optional[environment.Response] = None

    def set_policy(self, policy_: policy.Policy):
        self.policy = policy_

    def generate_episode(self) -> episode.Episode:
        episode_: episode.Episode = episode.Episode()
        # start
        response = self.environment.start()
        self.state = response.state

        while not self.state.is_terminal:
            self.action = self.policy.get_action_given_state(self.state)
            if self.verbose:
                print(f"state = {self.state} \t action = {self.action}")
            self.response = self.environment.apply_action_to_state(self.state, self.action)
            episode_.add_normal(self.state, self.action, self.response.reward)
            self.state = self.response.state
        episode_.add_terminal(self.state)

        return episode_
