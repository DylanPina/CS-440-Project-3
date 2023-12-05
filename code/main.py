import logging
from config import init_logging
from wire_diagram import WireDiagram, Wire, Seed, print_wire_diagram, WireDiagramDataLoader

if __name__ == '__main__':
    init_logging()

    # wire_diagram = WireDiagram()
    wire_diagram = Seed([(Wire.RED, 15, 1), (Wire.YELLOW, 0, 0), (Wire.GREEN, 11, 0),
                         (Wire.BLUE, 4, 0)])

    print_wire_diagram(wire_diagram.diagram)
    logging.debug(f"Wire placement: {wire_diagram.wire_placement}")
    logging.debug(
        f"Wire diagram dangerous (Red over Yellow): {wire_diagram.is_dangerous}")

    print(WireDiagramDataLoader(1, 0, 0).load_data())
