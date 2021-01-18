import numpy as np

import environment

rng: np.random.Generator = np.random.default_rng()


def environment_test() -> bool:
    racetrack_ = environment.track.RaceTrack(environment.track.TRACK_1, rng)
    environment_ = environment.Environment(racetrack_)

    for state_ in environment_.states():
        print(state_)

    print()

    for action_ in environment_.actions():
        print(action_)

    print()

    state_ = environment.State(0, 0, vx=0, vy=4)
    print(state_)
    for action_ in environment_.actions_for_state(state_):
        print(action_)

    return True


if __name__ == '__main__':
    if environment_test():
        print("Passed")
