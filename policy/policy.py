import abc

import state
import action


class Policy(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state_: state.State) -> action.Action:
        pass
