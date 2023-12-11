import numpy as np
import logging
from config import init_logging
from wire_diagram import WireDiagram, WireDiagramCell, Seed, print_wire_diagram, WireDiagramDataLoader
from networks import Network
from logistic_regression import LogisticRegression

if __name__ == '__main__':
    init_logging()

    training_data, validation_data, test_data = WireDiagramDataLoader(
        1000, 0, 100).load_data()

    logRegression = LogisticRegression(1600)
    logRegression.stochastic_gradient_descent(
        training_data, 0.01, 100, test_data)
