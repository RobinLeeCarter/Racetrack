from dataclasses import dataclass
from typing import Optional

import rsa

@dataclass
class RewardStateAction:
    reward: Optional[int]
    state: Optional[rsa.State]
    action: Optional[rsa.Action]

    @property
    def tuple(self) -> tuple:
        return self.reward, self.state, self.action
