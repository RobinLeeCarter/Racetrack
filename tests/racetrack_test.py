import numpy as np

import enums
from environment import track

rng: np.random.Generator = np.random.default_rng()


def racetrack_test1() -> bool:
    racetrack_ = track.RaceTrack(track.TRACK_1, rng)
    assert racetrack_.get_square(0, 0) == enums.Square.START
    assert racetrack_.get_square(5, 0) == enums.Square.GRASS
    assert racetrack_.get_square(5, 6) == enums.Square.END
    assert racetrack_.get_square(0, 6) == enums.Square.TRACK

    assert racetrack_.get_square(-1, -1) == enums.Square.GRASS
    assert racetrack_.get_square(6, 7) == enums.Square.GRASS
    assert racetrack_.get_square(6, 6) == enums.Square.END
    assert racetrack_.get_square(10, 6) == enums.Square.END
    assert racetrack_.get_square(-1, 6) == enums.Square.GRASS

    for _ in range(10):
        x, y = racetrack_.get_a_start_position()
        print(f"start position = ({x}, {y})")

    return True


if __name__ == '__main__':
    if racetrack_test1():
        print("Passed")
