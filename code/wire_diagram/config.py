from enum import Enum


class Wire(Enum):
    RED = 1
    BLUE = 2
    YELLOW = 3
    GREEN = 4

    def __str__(self):
        return "%s" % self.value
