import logging
import numpy as np
import random
from typing import List, Tuple


class LogisticRegression():

    def __init__(self, input: int):
        # Initialize weight matrices for each layer (expect the input layer)
        # self.weights = [np.random.randn(y, x)
        #                 for x, y in zip(self.layers[:-1], self.layers[1:])]
        self.weights = [np.random.randn() for _ in range(input)]
        # Initialize bias matrices for each layer (except the input layer)
        # self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]

    def feedforward(self, activation: np.ndarray, weights: np.ndarray) -> float:
        """
        Perform the feedforward operation in the network.

        This method computes the output of the network for a given input by propagating the input through each layer. 
        In each layer, it calculates the weighted sum of the inputs and biases, and then applies a sigmoid activation function.  This process is repeated for all layers in the network.
        """
        activation = activation.reshape(-1, 1)
        z = np.dot(weights, activation)
        # Apply sigmoid activation
        activation = self.sigmoid(z)
        # print(f"Feedforward: {activation}")
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
                # print(1)

                activation = input_data.reshape(-1, 1)
                z = np.dot(self.weights, activation)
                activation = self.sigmoid(z)

                logging.debug(f"activation: {activation}")
                # print(2)
                logging.debug(f"input dat: {input_data}")
                logging.debug(input_data.reshape(-1, 1).transpose())
                logging.debug(f"expected output: {expected_output}")
                logging.debug(
                    f"activation - expected output: {activation - expected_output}")

                grad_w = np.dot((activation - expected_output),
                                input_data.reshape(-1, 1).transpose())

                logging.debug(len(grad_w))

                # logging.debug(len(grad_w[0]))
                # logging.debug(len(grad_w[1]))
                # logging.debug(f"grad_w: {grad_w[0]}")
                # print(3)

                # self.weights = [w - learning_rate *
                #                 grad_w for w in self.weights]

                logging.debug(
                    f"learning rate * grad_w = {learning_rate * grad_w}")

                self.weights = self.weights - (learning_rate * grad_w)
                # print(4)

                # grad_w, grad_b = self.backpropagation(
                #     input_data, expected_output)
                # # Update the weights of the network by subtracting the gradient scaled by the learning rate
                # self.weights = [w - learning_rate *
                #                 gw for w, gw in zip(self.weights, grad_w)]
                # # Update the biases in a similar way, adjusting them based on their gradients and the learning rate
                # # self.biases = [b - learning_rate *
                # #                gb for b, gb in zip(self.biases, grad_b)]
            if test_data:
                logging.debug(
                    f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                logging.debug(f"Epoch {epoch} finished")

    def evaluate(self, test_data: np.ndarray) -> int:
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results = [(np.argmax(self.feedforward(x, self.weights)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(x == np.argmax(y) for (x, y) in test_results)

    def sigmoid(self, z: np.ndarray) -> float:
        """Sigmoid activation function."""

        return 1 / (1 + np.exp(-z))
