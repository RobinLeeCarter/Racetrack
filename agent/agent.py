from typing import Optional

import environment
import policy
import agent


class Agent:
    def __init__(self, environment_: environment.Environment, policy_: policy.Policy):
        self.environment: environment.Environment = environment_
        self.policy: policy.Policy = policy_
        self.action: Optional[environment.Action] = None
        self.state: Optional[environment.State] = None
        self.response: Optional[environment.Response] = None

    def set_policy(self, policy_: policy.Policy):
        self.policy = policy_

    def generate_episode(self) -> agent.Episode:
        episode_: agent.Episode = agent.Episode()
        # start
        response = self.environment.start()
        self.state = response.state

        while not self.state.is_terminal:
            self.action = self.policy.get_action_given_state(self.state)
            self.response = self.environment.apply_action(self.action)
            episode_.add(self.state, self.action, self.response.reward)
            self.state = self.response.state

        return episode_
