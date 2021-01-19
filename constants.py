import numpy as np
from environment.track import tracks

MIN_VELOCITY: int = 0
MAX_VELOCITY: int = 4

MIN_ACCELERATION: int = -1
MAX_ACCELERATION: int = +1

TRACK: np.ndarray = tracks.TRACK_2

INITIAL_Q_VALUE: float = -30.0
EXTRA_REWARD_FOR_FAILURE: float = -30.0   # 0.0 for problem statement

SKID_PROBABILITY: float = 0.1   # 0.1 for problem statement

LEARNING_EPISODES: int = 100_000
EPISODE_MEASUREMENT_FREQUENCY: int = 1_000
