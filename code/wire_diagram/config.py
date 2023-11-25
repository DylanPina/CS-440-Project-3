import logging
from enum import Enum
from typing import List, Optional


class Wire(Enum):
    RED = 1
    BLUE = 2
    YELLOW = 3
    GREEN = 4

def print_wire_diagram(wire_diagram: List[List[Optional[Wire]]]) -> None:
    """Outputs the wire diagram data to the log"""

    if not logging.DEBUG >= logging.root.level:
        return

    output = "\n--Wire Diagram--\n"
    for row in range(len(wire_diagram)):
        for col in range(len(wire_diagram)):
            output += f"{wire_diagram[row][col].value}, "

        output = output.rsplit(", ", 1)[0]
        if row != len(wire_diagram) - 1:
            output += "\n"

    logging.debug(output)