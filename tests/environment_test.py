import numpy as np

import racetrack
import tracks
import environment

rng: np.random.Generator = np.random.default_rng()


def environment_test() -> bool:
    racetrack_ = racetrack.RaceTrack(tracks.TRACK_1, rng)
    environment_ = environment.Environment(racetrack_)

    for state_ in environment_.states():
        print(state_)

    for action_ in environment_.actions():
        print(action_)

    return True


if __name__ == '__main__':
    if environment_test():
        print("Passed")
