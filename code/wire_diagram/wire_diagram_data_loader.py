import logging
import numpy as np
from .wire_diagram import WireDiagram
from typing import List, Tuple


class WireDiagramDataLoader():
    """Loads data for network to train, validate, and test on"""

    def __init__(self, training_data_count: int, validation_data_count: int, test_data_count: int):
        self.training_data_count = training_data_count
        self.validation_data_count = validation_data_count
        self.test_data_count = test_data_count

    def load_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns a tuple consisting of training_data, validation data, and test data"""

        training_data, validation_data, test_data = [], [], []

        for _ in range(self.training_data_count):
            wireDiagram = WireDiagram()
            x, y = np.array(
                wireDiagram.flatten_diagram()).reshape(-1), wireDiagram.is_dangerous
            training_data.append((x, y))

        for _ in range(self.validation_data_count):
            wireDiagram = WireDiagram()
            x, y = np.array(
                wireDiagram.flatten_diagram()).reshape(-1), wireDiagram.is_dangerous
            validation_data.append((x, y))

        for _ in range(self.test_data_count):
            wireDiagram = WireDiagram()
            x, y = np.array(
                wireDiagram.flatten_diagram()).reshape(-1), wireDiagram.is_dangerous
            test_data.append((x, y))

        return (training_data, validation_data, test_data)
