from dataclasses import dataclass
from typing import Optional

import state
import action


@dataclass
class RewardStateAction:
    reward: Optional[int]
    state: Optional[state.State]
    action: Optional[action.Action]

    @property
    def tuple(self) -> tuple:
        return self.reward, self.state, self.action
