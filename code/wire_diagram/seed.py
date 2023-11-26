from typing import List, Optional, Tuple
from .config import Wire
from .utils import init_diagram, place_row, place_col, classify_diagram


class Seed():
    def __init__(self, wire_placement: List[Tuple[Wire, int]]):
        self.wire_placement = wire_placement  # (Wire, row/col, direction)
        self.diagram = self.create_diagram()
        self.is_dangerous = classify_diagram(self.wire_placement)

    def create_diagram(self) -> List[List[Optional[Wire]]]:
        """Returns a wire diagram given the wire placements"""

        diagram = init_diagram()
        for wire, x, direction in self.wire_placement:
            if direction:
                place_row(diagram, x, wire)
            else:
                place_col(diagram, x, wire)
        return diagram
