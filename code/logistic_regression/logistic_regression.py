import logging
import numpy as np
import random
from typing import List, Tuple
from wire_diagram import utils


class LogisticRegression():

    def __init__(self, input: int):
        # Initialize weight matrices for each layer (expect the input layer)
        self.weights = [np.random.randn() for _ in range(input)]

    def feedforward(self, activation: np.ndarray) -> float:
        """
        Perform the feedforward operation in the network.

        This method computes the output of the network for a given input by propagating the input through each layer. 
        In each layer, it calculates the weighted sum of the inputs and biases, and then applies a sigmoid activation function.  This process is repeated for all layers in the network.
        """

        activation = activation.reshape(-1, 1)
        z = np.dot(self.weights, activation)
        activation = self.sigmoid(z)
        return activation

    def stochastic_gradient_descent(
        self,
        training_data: Tuple[np.ndarray, np.ndarray],
        learning_rate: float,
        epochs: int,
        test_data: Tuple[np.ndarray, np.ndarray] = None
    ) -> None:
        """
        Perform Stochastic Gradient Descent (SGD) to optimize the network.

        This function updates the weights and biases of the network based on the gradients computed
        from each training sample or a small batch of samples.
        """

        # Loop over each epoch (an epoch is one complete pass through the entire training set)
        for epoch in range(epochs):
            # print(f"epoch: {epoch}")
            random.shuffle(training_data)

            # Iterate over each example in the training data
            for input_data, expected_output in training_data:
                activation = self.feedforward(input_data)

                # logging.debug(f"activation: {activation}")
                # logging.debug(f"expected output: {expected_output}")

                grad_w = np.dot((activation - expected_output),
                                input_data.reshape(-1, 1).transpose())

                self.weights -= (learning_rate * grad_w)
                logging.info(
                    f"{self.evaluate(training_data)} / {len(training_data)}")

            # if test_data:
            #     logging.info(
            #         f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            # else:
            logging.info(f"Epoch {epoch} finished")

    def evaluate(self, test_data: np.ndarray) -> int:
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results = [(utils.evaluate_activation(self.feedforward(x)), y)
                        for (x, y) in test_data]

        return sum(x == y for (x, y) in test_results)

    def sigmoid(self, z: np.ndarray) -> float:
        """Sigmoid activation function."""

        return 1 / (1 + np.exp(-z))
