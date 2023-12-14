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

    def generate_non_linear_features(self, diagram: np.ndarray, t1: bool = False, t2: bool = False) -> np.ndarray:
        """Returns non-linear features for the specified task."""

        if t1:
            return self.generate_non_linear_features_t1(diagram, (2, 2), t1, t2)
        elif t2:
            return self.generate_non_linear_features_t2(diagram, (2, 2), t1, t2)

        return np.array([None])

    def generate_non_linear_features_t1(self, diagram: np.ndarray, region_size: Tuple[int] = (2, 2), t1: bool = False, t2: bool = False) -> np.ndarray:
        """
        Generate convolutional-like features from a flattened wiring diagram.

        It extracts local features from a 20x20 pixel wiring diagram, which is represented 
        as a flattened one-hot encoded vector of length 1600. The function reshapes this vector
        into a 20x20x4 grid (for the one-hot encoded color channels) and extracts features from 
        local regions of specified size by summing the occurrences of each color.
        """

        diagram_matrix = diagram.reshape(20, 20, 4)

        features = []
        for i in range(0, 20 - region_size[0] + 1):
            for j in range(0, 20 - region_size[1] + 1):
                region = diagram_matrix[i:i +
                                        region_size[0], j:j+region_size[1], :]
                region_features = np.sum(region, axis=(0, 1))
                features.append(region_features)

        return np.array(features).flatten()

    def generate_non_linear_features_t2(self, diagram: np.ndarray, region_size: Tuple[int] = (2, 2), t1: bool = False, t2: bool = False) -> np.ndarray:
        """
        Generate features based on the intersections and sequence of wire colors in each region.
        
        It analyzes each region to check for the presence of each color. It marks intersections 
        of each color within the region. The feature vector indicates which colors are intersecting 
        in each local region.
        """

        diagram_matrix = diagram.reshape(20, 20, 4)

        features = []
        for i in range(0, 20 - region_size[0] + 1, 1):
            for j in range(0, 20 - region_size[1] + 1, 1):
                region = diagram_matrix[i:i +
                                        region_size[0], j:j+region_size[1], :]
                intersection_features = np.zeros(4)
                # Analyze the layering of colors
                for color_index in range(4):
                    if np.any(region[:, :, color_index]):
                        # Mark if this color intersects in the region
                        intersection_features[color_index] = 1
                features.append(intersection_features)

        return np.array(features).flatten()

    def flatten_diagram_non_linear(self, t1: bool = False, t2: bool = False) -> np.ndarray:
        """Flattens the wires into binary values with non linear featrues"""

        flattened_diagram = self.flatten_diagram()
        non_linear_features = self.generate_non_linear_features(
            flattened_diagram, t1, t2)
        flattened_diagram_non_linear = np.concatenate(
            (flattened_diagram, non_linear_features))

        return flattened_diagram_non_linear
