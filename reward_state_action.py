from dataclasses import dataclass
from typing import Optional

import state
import action


@dataclass
class RewardStateAction:
    reward: Optional[int]
    state: Optional[state.State]
    action: Optional[action.Action]
