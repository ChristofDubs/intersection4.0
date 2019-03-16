from enum import IntEnum


class SectionIndex(IntEnum):
    BEFORE_INTERSECTION = 0
    IN_INTERSECTION_STRAIGHT = 1
    AFTER_INTERSECTION = 2
    IN_INTERSECTION_RIGHT = 3
    IN_INTERSECTION_LEFT = 4
    NUM_SEGMENTS = 5


class Target(IntEnum):
    GO_STRAIGHT = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2


class Action(IntEnum):
    KEEP_SPEED = 0
    ACCELERATE = 1
    DECELERATE = 2
