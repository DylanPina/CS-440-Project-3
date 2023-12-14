import logging
import math
import numpy as np
import random
import utils
from typing import List, Tuple


class LogisticRegression():

    def __init__(
        self,
        input_layer_size: int,
        training_data: Tuple[np.ndarray, np.ndarray],
        testing_data: Tuple[np.ndarray, np.ndarray],
        learning_rate: float,
        epochs: int,
        lambda_term: float,
        patience: int
    ):
        self.input_layer_size = input_layer_size
        self.training_data = training_data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_term = lambda_term
        self.patience = patience
        self.testing_data = testing_data
        self.weights = [np.random.randn() for _ in range(input_layer_size)]
        self.training_loss = []
        self.testing_loss = []
        self.training_success = []
        self.testing_success = []
        # (min testing loss, weights for that epoch)
        self.min_testing_loss = (float("inf"), [])
        self.stop_training = False

    def feedforward(self, activation: np.ndarray) -> float:
        """Perform the feedforward operation in the network."""

        activation = activation.reshape(-1, 1)
        z = np.dot(self.weights, activation)
        activation = self.sigmoid(z)
        return activation

    def stochastic_gradient_descent(self) -> None:
        """Perform Stochastic Gradient Descent (SGD) to optimize the model."""

        for epoch in range(self.epochs):
            if self.stop_training:
                logging.info(f"Stopped training at Epoch: {epoch}")
                break

            random.shuffle(self.training_data)

            # Iterate over each example in the training data
            for input_data, expected_output in self.training_data:
                activation = self.feedforward(input_data)
                grad_w = self.gradient_descent(
                    activation, input_data, expected_output)
                self.weights -= (self.learning_rate * grad_w)

            self.calculate_loss()

    def gradient_descent(self, activation: float, input_data: np.ndarray, expected_output: int) -> np.ndarray:
        """Compute the gradient of the loss function."""

        return np.dot((activation - expected_output),
                      input_data.reshape(-1, 1).transpose())

    def gradient_descent_l2_regularization(self, activation: float, input_data: np.ndarray, expected_output: int) -> np.ndarray:
        """Compute the gradient of the loss function with L2 regularization."""

        gradient = np.dot((activation - expected_output),
                          input_data.reshape(-1, 1).transpose())
        l2_gradient = np.dot(self.lambda_term, self.weights)
        return gradient + l2_gradient

    def binary_cross_entropy_loss(self, data: List[Tuple[np.ndarray]]) -> float:
        """Calculate the binary cross entropy loss."""

        total_loss = 0
        for x, y in data:
            # To avoid log(0) which is undefined, small values are added/subtracted from p
            x = max(min(self.feedforward(x), 1 - 1e-15), 1e-15)
            total_loss += y * math.log(x) + (1 - y) * math.log(1 - x)

        return -total_loss / len(data)

    def l2_regularization(self) -> float:
        """Calculate the l2 regularization."""

        return self.lambda_term * np.sum(np.square(self.weights))

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

        training_loss = self.binary_cross_entropy_loss(
            self.training_data)
        testing_loss = self.binary_cross_entropy_loss(
            self.testing_data)

        self.terminate_early(testing_loss)

        training_success = self.evaluate(
            self.training_data) / len(self.training_data)
        testing_success = self.evaluate(
            self.testing_data) / len(self.testing_data)

        self.training_success.append(training_success)
        self.testing_success.append(testing_success)

        self.training_loss.append(training_loss)
        self.testing_loss.append(testing_loss)

    def terminate_early(self, current_testing_loss: float) -> bool:
        """Return True if the model's performance is plateauing."""

        if current_testing_loss < self.min_testing_loss[0]:
            self.min_testing_loss = (current_testing_loss, self.weights)
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True

    def plot_loss(self, linear: bool) -> None:
        """Plots the loss of the training and testing data sets"""

        utils.plot(f"training_loss_{f'linear' if linear else 'non-linear'}_d-{len(self.training_data)}_e-{self.epochs}_a-{self.learning_rate}", f"Training Data Loss {f'Linear' if linear else 'Non-linear'} - Data size: {len(self.training_data)} - Epochs: {self.epochs} - Alpha: {self.learning_rate}",
                   "Epochs", "Loss", self.training_loss)
        utils.plot(f"testing_loss_{f'linear' if linear else 'non-linear'}_d-{len(self.testing_data)}_e-{self.epochs}_a-{self.learning_rate}", f"Testing Data Loss {f'Linear' if linear else 'Non-linear'} - Data size: {len(self.testing_data)} - Epochs: {self.epochs} - Alpha: {self.learning_rate}",
                   "Epochs", "Loss", self.testing_loss)

    def plot_success(self, linear: bool) -> None:
        """Plots the success rate of the training and testing data sets"""

        utils.plot(f"training_success_{f'linear' if linear else 'non-linear'}_d-{len(self.training_data)}_e-{self.epochs}_a-{self.learning_rate}", f"Training Success {f'Linear' if linear else 'Non-linear'} - Data size: {len(self.training_data)} - Epochs: {self.epochs} - Alpha: {self.learning_rate}", "Epochs", "Success Rate",
                   self.training_success)
        utils.plot(f"testing_success_{f'linear' if linear else 'non-linear'}_d-{len(self.testing_data)}_e-{self.epochs}_a-{self.learning_rate}", f"Testing Success {f'Linear' if linear else 'Non-linear'} - Data size: {len(self.testing_data)} - Epochs: {self.epochs} - Alpha: {self.learning_rate}",
                   "Epochs", "Success Rate", self.testing_success)
