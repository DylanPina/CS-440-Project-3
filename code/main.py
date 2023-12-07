import numpy as np
import logging
from config import init_logging
from wire_diagram import WireDiagram, WireDiagramCell, Seed, print_wire_diagram, WireDiagramDataLoader
from networks import Network

if __name__ == '__main__':
    init_logging()

    # wire_diagram = WireDiagram()
    # wire_diagram = Seed([(Wire.RED, 15, 1), (Wire.YELLOW, 0, 0), (Wire.GREEN, 11, 0),
    #                      (Wire.BLUE, 4, 0)])

    # print_wire_diagram(wire_diagram.diagram)
    # logging.debug(f"Wire placement: {wire_diagram.wire_placement}")
    # logging.debug(
    #     f"Wire diagram dangerous (Red over Yellow): {wire_diagram.is_dangerous}")

    training_data, validation_data, test_data = WireDiagramDataLoader(
        100, 100, 100).load_data()
    # print(validation_data)
    # print(test_data)

    network = Network([1600, 10, 2])
    network.stochastic_gradient_descent(training_data, 0.01, 100, test_data)
