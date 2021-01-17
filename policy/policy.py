import abc

import environment


class Policy(abc.ABC):
    def __init__(self, environment_: environment.Environment):
        self.environment = environment_

    def get_next_action(self) -> environment.Action:
        return self.get_action_given_state(self.environment.state)

    @abc.abstractmethod
    def get_action_given_state(self, state_: environment.State) -> environment.Action:
        pass

    @abc.abstractmethod
    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        pass
