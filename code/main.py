from config import init_logging
from wire_diagram import WireDiagram, print_wire_diagram

if __name__ == '__main__':
    init_logging()

    wire_diagram = WireDiagram()
    print_wire_diagram(wire_diagram.diagram)