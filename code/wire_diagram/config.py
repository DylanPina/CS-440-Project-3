from enum import Enum


class WireDiagramCell(Enum):
    NO_WIRE = [0, 0, 0, 0, 1]
    RED = [0, 0, 0, 1, 0]
    BLUE = [0, 0, 1, 0, 0]
    YELLOW = [0, 1, 0, 0, 0]
    GREEN = [1, 0, 0, 0, 0]

    def __str__(self):
        return "%s" % self.value
