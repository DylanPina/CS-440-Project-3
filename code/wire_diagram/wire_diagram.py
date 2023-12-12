import numpy as np
import logging
import random
from typing import List, Tuple
from .config import WireDiagramCell
from .utils import init_diagram, classify_diagram, place_row, place_col


class WireDiagram:
    def __init__(self):
        self.diagram = init_diagram()
        self.is_dangerous = False

    def place_wires(self) -> List[Tuple[WireDiagramCell, int]]:
        """
        Places the wires on the diagram and returns if the diagram is classified as dangerous
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

        return classify_diagram(placement)

    def place_wires_dangerously(self) -> Tuple[List[Tuple[WireDiagramCell, int, int]], WireDiagramCell]:
        """
        Places the wires on the diagram in a dangerous configuration and returns which wire to cut
        """

        # Ensure Red is placed before Yellow
        wire_order = [WireDiagramCell.RED, WireDiagramCell.YELLOW,
                      WireDiagramCell.BLUE, WireDiagramCell.GREEN]
        random.shuffle(wire_order)
        red_index = wire_order.index(WireDiagramCell.RED)
        yellow_index = wire_order.index(WireDiagramCell.YELLOW)

        # If Yellow comes before Red, swap them
        if yellow_index < red_index:
            wire_order[red_index], wire_order[yellow_index] = wire_order[yellow_index], wire_order[red_index]

        direction = 1 if random.random() > 0.5 else 0
        remaining_cols, remaining_rows = list(range(20)), list(range(20))
        placement: List[Tuple] = []  # (Wire, row/col, direction)

        for wire in wire_order:
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

        # The wire to cut is the third one placed
        wire_to_cut = placement[2][0]

        return wire_to_cut

    def flatten_diagram(self) -> np.ndarray:
        """Flattens the wires into binary values"""

        return np.array([cell.value for row in self.diagram for cell in row]).flatten()

    def generate_non_linear_features(self, diagram: np.ndarray) -> np.ndarray:
        """
        Generates non-linear features based on horizontal and vertical adjacent cells.

        In each row of 20 cells, there are 19 interactions (between the first and second
        cell, second and third, and so onup to the 19th and 20th). Since there are 20 rows,
        the total number of horizontal interactions is 19 * 20 = 380.

        Similarly, in each column of 20 cells, there are also 19 interactions (between the
        first and second cell, second and third, etc., down the column). With 20 columns,
        the total number of vertical interactions is also 19 * 20 = 380.

        Adding horizontal and vertical interactions produce a total of 760 new features.
        """

        features = []
        num_rows = num_columns = 20  # 20 x 20 grid

        # Horizontal Interactions (Adjacent cells in a row)
        for row in range(num_rows):
            # Loop to second-to-last cell to avoid index out of bounds
            for i in range(num_rows - 1):
                index1 = row * num_rows + i
                index2 = index1 + 1  # Adjacent cell
                interaction = diagram[index1] * diagram[index2]
                features.append(interaction)

        # Vertical Interactions (Adjacent cells in a column)
        for col in range(num_columns):
            # Loop to second-to-last cell to avoid index out of bounds
            for i in range(num_rows - 1):
                index1 = col + i * num_columns
                # Cell in the next row (directly below)
                index2 = index1 + num_columns
                interaction = diagram[index1] * diagram[index2]
                features.append(interaction)

        return np.array(features)

    def flatten_diagram_non_linear(self) -> np.ndarray:
        """Flattens the wires into binary values with non linear featrues"""

        flattened_diagram = self.flatten_diagram()
        non_linear_features = self.generate_non_linear_features(
            flattened_diagram)
        flattened_diagram_non_linear = np.concatenate(
            (flattened_diagram, non_linear_features))

        return flattened_diagram_non_linear
