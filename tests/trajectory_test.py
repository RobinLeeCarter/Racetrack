import numpy as np

import racetrack
import tracks
from episode import trajectory
import policy

rng: np.random.Generator = np.random.default_rng()


def trajectory_test() -> bool:
    racetrack_ = racetrack.RaceTrack(tracks.TRACK_1, rng)
    behaviour_ = policy.RandomPolicy(rng, actions_shape=(3, 3))
    trajectory_ = trajectory.Trajectory(racetrack_, verbose=True)

    # a = action.Action(ax=1, ay=1)
    t = 0
    while not trajectory_.is_terminated:
        a = behaviour_.get_action(trajectory_.current.state)
        print(f"t={t} \t state = {trajectory_.current.state} \t action = {a}")
        trajectory_.apply_action(a)
        t += 1
    trajectory_.output()
    print(f"final position = {trajectory_.current.state.x}, {trajectory_.current.state.y}")

    return True


if __name__ == '__main__':
    if trajectory_test():
        print("Passed")
