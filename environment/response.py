from __future__ import annotations
from dataclasses import dataclass

from environment import state


@dataclass
class Response:
    state: state.State
    reward: float
