import enum


class Square(enum.IntEnum):
    TRACK = 0
    GRASS = 1
    START = 2
    END = 3
    CAR = 4


class UserEvent(enum.IntEnum):
    NONE = 0
    QUIT = 1
    SPACE = 2
