import numpy as np
from environment.track import tracks

MIN_VELOCITY: int = 0
MAX_VELOCITY: int = 4

MIN_ACCELERATION: int = -1
MAX_ACCELERATION: int = +1

TRACK: np.ndarray = tracks.TRACK_3

INITIAL_Q_VALUE: float = -30.0

SKID_PROBABILITY: float = 0.1
