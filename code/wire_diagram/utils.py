import logging
from typing import List, Optional
from .config import Wire


def init_diagram() -> List[List[Optional[Wire]]]:
    """Initializes the wire diagram to an empty 20x20 matrix"""

    return [[None] * 20 for _ in range(20)]


def print_wire_diagram(wire_diagram: List[List[Optional[Wire]]], msg: str = None) -> None:
    """Outputs the wire diagram data to the log"""

    if not logging.DEBUG >= logging.root.level:
        return

    output = f"\n--Wire Diagram{(' ' + msg) if msg else ''}--\n"
    for row in range(len(wire_diagram)):
        for col in range(len(wire_diagram)):
            curr = wire_diagram[row][col]
            output += f"{curr.value if curr else 0}, "

        output = output.rsplit(", ", 1)[0]
        if row != len(wire_diagram) - 1:
            output += "\n"
    
    logging.debug(output)


def place_row(diagram: List[List[Optional[Wire]]], row: int, wire: Wire) -> None:
    """Places wire on a specified row"""

    # Set every element in the specified row to the given value
    diagram[row] = [wire] * 20


def place_col(diagram: List[List[Optional[Wire]]], col: int, wire: Wire) -> None:
    """Places wire on a specified row"""

    for row in diagram:
        row[col] = wire


def classify_diagram(wire_placement: List[Wire]) -> bool:
    """Returns True if the diagram is classified as dangerous"""

    red_wire_index = next((i for i, (wire, _, _) in enumerate(
        wire_placement) if wire == Wire.RED), -1)
    yellow_wire_index = next((i for i, (wire, _, _) in enumerate(
        wire_placement) if wire == Wire.YELLOW), -1)

    red_wire_direction = wire_placement[red_wire_index][2]
    yellow_wire_direction = wire_placement[yellow_wire_index][2]

    return (red_wire_index < yellow_wire_index) and (red_wire_direction != yellow_wire_direction)
