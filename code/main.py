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
        1, 1, 1).load_data()
    print(training_data)
    print(validation_data)
    print(test_data)
    network = Network([600, 10, 2])
