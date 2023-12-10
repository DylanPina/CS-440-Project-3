import numpy as np
import logging
from typing import List, Optional
from .config import WireDiagramCell


def init_diagram() -> List[List[Optional[WireDiagramCell]]]:
    """Initializes the wire diagram to an empty 20x20 matrix"""

    return [[WireDiagramCell.NO_WIRE] * 20 for _ in range(20)]


def print_wire_diagram(wire_diagram: List[List[Optional[WireDiagramCell]]], msg: str = None) -> None:
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


def place_row(diagram: List[List[Optional[WireDiagramCell]]], row: int, wire: WireDiagramCell) -> None:
    """Places wire on a specified row"""

    # Set every element in the specified row to the given value
    diagram[row] = [wire] * 20


def place_col(diagram: List[List[Optional[WireDiagramCell]]], col: int, wire: WireDiagramCell) -> None:
    """Places wire on a specified row"""

    for row in diagram:
        row[col] = wire


def classify_diagram(wire_placement: List[WireDiagramCell]) -> np.ndarray:
    """Returns one-hot encoding for diagram's wire placement"""

    red_wire_index = next((i for i, (wire, _, _) in enumerate(
        wire_placement) if wire == WireDiagramCell.RED), -1)
    yellow_wire_index = next((i for i, (wire, _, _) in enumerate(
        wire_placement) if wire == WireDiagramCell.YELLOW), -1)

    red_wire_direction = wire_placement[red_wire_index][2]
    yellow_wire_direction = wire_placement[yellow_wire_index][2]

    is_dangerous = (red_wire_index < yellow_wire_index) and (
        red_wire_direction != yellow_wire_direction)

    return one_hot_encode(is_dangerous)


def one_hot_encode(is_dangerous: bool) -> np.ndarray:
    """Returns one-hot encoding for whether a diagram is classifed as dangerous or not dangerous"""

    # Define the one-hot encoding for 'dangerous' and 'not dangerous'
    # One-hot encoding for 'dangerous'
    encoding_dangerous = np.array([[1], [0]])
    # One-hot encoding for 'not dangerous'
    encoding_not_dangerous = np.array([[0], [1]])

    # Return the appropriate encoding based on the input
    return encoding_dangerous if is_dangerous else encoding_not_dangerous
