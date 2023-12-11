import math
import numpy as np
import random
import utils
from typing import List, Tuple


class LogisticRegression():

    def __init__(self, input_layer_size: int, training_data: Tuple[np.ndarray, np.ndarray], testing_data: Tuple[np.ndarray, np.ndarray], learning_rate: float, epochs: int):
        self.input_layer_size = input_layer_size
        self.training_data = training_data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.testing_data = testing_data
        self.weights = [np.random.randn() for _ in range(input_layer_size)]
        self.training_loss = []
        self.testing_loss = []
        self.training_success = []
        self.testing_success = []

    def feedforward(self, activation: np.ndarray) -> float:
        """Perform the feedforward operation in the network."""

        activation = activation.reshape(-1, 1)
        z = np.dot(self.weights, activation)
        activation = self.sigmoid(z)
        return activation

    def stochastic_gradient_descent(self) -> None:
        """Perform Stochastic Gradient Descent (SGD) to optimize the model."""

        for _ in range(self.epochs):
            random.shuffle(self.training_data)

            # Iterate over each example in the training data
            for input_data, expected_output in self.training_data:
                activation = self.feedforward(input_data)
                grad_w = np.dot((activation - expected_output),
                                input_data.reshape(-1, 1).transpose())
                self.weights -= (self.learning_rate * grad_w)

            self.calculate_loss()

    def binary_cross_entropy_loss(self, data: List[Tuple[np.ndarray]]) -> float:
        """Calculate the binary cross entropy loss."""

        total_loss = 0
        for x, y in data:
            # To avoid log(0) which is undefined, small values are added/subtracted from p
            x = max(min(self.feedforward(x), 1 - 1e-15), 1e-15)
            total_loss += y * math.log(x) + (1 - y) * math.log(1 - x)

        return -total_loss / len(data)

    def evaluate(self, data: List[Tuple[np.ndarray]]) -> int:
        """Return the number of test inputs for which the model network outputs the correct result."""

        test_results = [(utils.evaluate_activation(self.feedforward(x)), y)
                        for (x, y) in data]

        return sum(x == y for (x, y) in test_results)

    def sigmoid(self, z: np.ndarray) -> float:
        """Sigmoid activation function."""

        return 1 / (1 + np.exp(-z))

    def calculate_loss(self) -> None:
        """Calculates the loss for training and testing data based on the current model"""

        training_loss = self.binary_cross_entropy_loss(self.training_data)
        testing_loss = self.binary_cross_entropy_loss(self.testing_data)

        training_success = self.evaluate(
            self.training_data) / len(self.training_data)
        testing_success = self.evaluate(
            self.testing_data) / len(self.testing_data)

        self.training_success.append(training_success)
        self.testing_success.append(testing_success)

        self.training_loss.append(training_loss)
        self.testing_loss.append(testing_loss)

    def plot_loss(self) -> None:
        """Plots the loss of the training and testing data sets"""

        utils.plot(f"training_loss_d-{len(self.training_data)}_e-{self.epochs}_a-{self.learning_rate}", f"Training Data Loss - Data size: {len(self.training_data)} - Epochs: {self.epochs} - Alpha: {self.learning_rate}",
                   "Epochs", "Loss", self.training_loss)
        utils.plot(f"testing_loss_d-{len(self.testing_data)}_e-{self.epochs}_a-{self.learning_rate}", f"Testing Data Loss - Data size: {len(self.testing_data)} - Epochs: {self.epochs} - Alpha: {self.learning_rate}",
                   "Epochs", "Loss", self.testing_loss)

    def plot_success(self) -> None:
        """Plots the success rate of the training and testing data sets"""

        utils.plot(f"training_success_d-{len(self.training_data)}_e-{self.epochs}_a-{self.learning_rate}", f"Training Success - Data size: {len(self.training_data)} - Epochs: {self.epochs} - Alpha: {self.learning_rate}", "Epochs", "Success Rate",
                   self.training_success)
        utils.plot(f"testing_success_d-{len(self.testing_data)}_e-{self.epochs}_a-{self.learning_rate}", f"Testing Success - Data size: {len(self.testing_data)} - Epochs: {self.epochs} - Alpha: {self.learning_rate}",
                   "Epochs", "Success Rate", self.testing_success)
