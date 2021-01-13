import numpy as np

import enums


class RaceTrack:
    def __init__(self, track: np.ndarray, rng: np.random.Generator):
        self.rng: np.random.Generator = rng
        self.track = track
        self.max_y = track.shape[0]
        self.max_x = track.shape[1]
        self.end_y = (self.track[:, self.max_x] == enums.Square.END)
        self.starts = (self.track[:, :] == enums.Square.START)
        self.starts_flat = np.flatnonzero(self.starts)

    def get_square(self, x: int, y: int) -> enums.Square:
        x_inside = (0 <= x < self.max_x)
        y_inside = (0 <= y < self.max_y)

        if y_inside and self.end_y[y] and x >= self.max_x:
            # over finish line
            return enums.Square.END
        elif not x_inside or not y_inside:
            # crash outside of track and not over finish line
            return enums.Square.GRASS
        else:
            # just whatever the track value is
            value = self.track[self.max_y - y, x]
            return enums.Square(value)

    def draw_start_position(self) -> tuple:
        start_flat = self.rng.choice(self.starts_flat)
        iy, ix = np.unravel_index(start_flat, shape=self.track.shape)
        y = self.max_y - iy
        x = ix
        return x, y
