import numpy as np

# import enums
import racetrack
import tracks
import trajectory
import policy

# import action

rng: np.random.Generator = np.random.default_rng()


def trajectory_test() -> bool:
    racetrack_ = racetrack.RaceTrack(tracks.TRACK_1, rng)
    behaviour_ = policy.RandomPolicy(rng)
    trajectory_ = trajectory.Trajectory(racetrack_, verbose=True)

    # a = action.Action(ax=1, ay=1)
    t = 0
    while t <= 10 and not trajectory_.is_terminated:
        a = behaviour_.get_action(trajectory_.current.state)
        print(f"t={t} \t state = {trajectory_.current.state} \t action = {a}")
        trajectory_.apply_action(a)
        t += 1

    return True


if __name__ == '__main__':
    if trajectory_test():
        print("Passed")
