from config import init_logging
from wire_diagram import WireDiagram, print_wire_diagram
import logging

if __name__ == '__main__':
    init_logging()

    wire_diagram = WireDiagram()
    print_wire_diagram(wire_diagram.diagram)
    logging.debug(
        f"Wire diagram dangerous (Red over Yellow): {wire_diagram.is_dangerous}")
    logging.debug(
        f"Wire placement (wire, direction): {wire_diagram.wire_placement}")
