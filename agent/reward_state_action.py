from dataclasses import dataclass
from typing import Optional

import environment


@dataclass
class RewardStateAction:
    reward: Optional[float]
    state: Optional[environment.State]
    action: Optional[environment.Action]

    @property
    def tuple(self) -> tuple:
        return self.reward, self.state, self.action
