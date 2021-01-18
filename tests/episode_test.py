import numpy as np

import environment
import agent
import policy

rng: np.random.Generator = np.random.default_rng()


def episode_test() -> bool:
    racetrack_ = environment.track.RaceTrack(environment.track.TRACK_1, rng)
    environment_ = environment.Environment(racetrack_, verbose=True)
    policy_ = policy.RandomPolicy(environment_, rng)
    agent_ = agent.Agent(environment_, policy_, verbose=True)
    episode_: agent.Episode = agent_.generate_episode()

    print()
    for t, rsa in enumerate(episode_.trajectory):
        print(f"t={t}\treward={rsa.reward}\tstate={rsa.state}\taction={rsa.action}")

    return True


if __name__ == '__main__':
    if episode_test():
        print("Passed")
