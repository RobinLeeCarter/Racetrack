from dataclasses import dataclass
from typing import Optional

import state
import action


@dataclass
class StateActionReward:
    state: state.State
    action: Optional[action.Action] = None
    reward: Optional[int] = None
