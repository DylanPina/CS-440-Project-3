import random
from typing import List, Optional, Tuple
from .config import Wire
import logging


class WireDiagram:
    def __init__(self):
        self.diagram = self.init_diagram()
        self.wire_placement = self.place_wires()
        self.is_dangerous = self.classify_diagram()

    def init_diagram(self) -> List[List[Optional[Wire]]]:
        """Initializes the wire diagram to an empty 20x20 matrix"""

        return [[Wire.BLANK] * 20 for _ in range(20)]

    def place_wires(self) -> List[Tuple[Wire, int]]:
        """
        Places the wires on the diagram and returns the order in which the wires were placed
        """

        direction = 1 if random.random() > 0.5 else 0
        remaining_wires = [wire for wire in Wire]
        remaining_wires.remove(Wire.BLANK)
        remaining_cols, remaining_rows = list(range(20)), list(range(20))
        placement: List[Tuple] = []  # (Wire, direction)

        while remaining_wires:
            wire = random.choice(remaining_wires)
            remaining_wires.remove(wire)

            if direction:
                row = random.choice(remaining_rows)
                remaining_rows.remove(row)
                self.place_row(row, wire)
            else:
                col = random.choice(remaining_cols)
                remaining_cols.remove(col)
                self.place_col(col, wire)

            placement.append((wire.value, direction))
            direction = 1 - direction

        return placement

    def place_row(self, row: int, wire: Wire) -> None:
        """Places wire on a specified row"""
        logging.debug(f"Row/wire_color: {row}/{wire.value}")
        for i in range(20):
            if i == row:
                for j in range(20):
                    self.diagram[i][j] = wire

    def place_col(self, col: int, wire: Wire) -> None:
        """Places wire on a specified row"""
        logging.debug(f"Col/wire_color {col}/{wire.value}")
        for i in range(20):
            for j in range(20):
                if j == col:
                    self.diagram[i][j] = wire

    def classify_diagram(self) -> bool:
        """Returns True if the diagram is classified as dangerous"""

        red_wire_index = next((i for i, (color, _) in enumerate(
            self.wire_placement) if color == Wire.RED.value), -1)
        yellow_wire_index = next((i for i, (color, _) in enumerate(
            self.wire_placement) if color == Wire.YELLOW.value), -1)

        red_wire_direction = self.wire_placement[red_wire_index][1]
        yellow_wire_direction = self.wire_placement[yellow_wire_index][1]

        return (red_wire_index < yellow_wire_index) and (red_wire_direction != yellow_wire_direction)
