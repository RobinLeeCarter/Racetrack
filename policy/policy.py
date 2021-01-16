import abc

from rsa import action, state


class Policy(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state_: state.State) -> action.Action:
        pass

    def get_probability(self, action_: action.Action, state_: state.State) -> float:
        pass
