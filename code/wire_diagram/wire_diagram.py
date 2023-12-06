import numpy as np
import logging
import random
from typing import List, Tuple
from .config import WireDiagramCell
from .utils import init_diagram, classify_diagram, place_row, place_col


class WireDiagram:
    def __init__(self):
        self.diagram = init_diagram()
        self.wire_placement = self.place_wires()  # (Wire, row/col, direction)
        self.is_dangerous = classify_diagram(self.wire_placement)

    def place_wires(self) -> List[Tuple[WireDiagramCell, int]]:
        """
        Places the wires on the diagram and returns the order in which the wires were placed
        """

        direction = 1 if random.random() > 0.5 else 0
        remaining_wires = [WireDiagramCell.RED, WireDiagramCell.BLUE,
                           WireDiagramCell.GREEN, WireDiagramCell.YELLOW]
        remaining_cols, remaining_rows = list(range(20)), list(range(20))
        placement: List[Tuple] = []  # (Wire, row/col, direction)

        while remaining_wires:
            wire = random.choice(remaining_wires)
            remaining_wires.remove(wire)

            if direction:
                row = random.choice(remaining_rows)
                remaining_rows.remove(row)
                place_row(self.diagram, row, wire)
                placement.append((wire, row, direction))
            else:
                col = random.choice(remaining_cols)
                remaining_cols.remove(col)
                place_col(self.diagram, col, wire)
                placement.append((wire, col, direction))

            direction = 1 - direction

        return placement

    def flatten_diagram(self) -> np.ndarray:
        """Flattens the wires into binary values"""

        return np.array([cell.value for row in self.diagram for cell in row]).flatten()
