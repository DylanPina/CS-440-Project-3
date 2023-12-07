import logging
import numpy as np
import random
from typing import List, Tuple


class Network():

    def __init__(self, layers: List[int]):
        # Represents the layers in the network (e.g., [1600, 10, 2] corresponds to 1600 input layer, 10 hidden layer, 2 output layer)
        self.layers = layers
        # Initialize weight matrices for each layer (expect the input layer)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.layers[:-1], self.layers[1:])]
        # Initialize bias matrices for each layer (except the input layer)
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]

    def feedforward(self, activation: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> float:
        """
        Perform the feedforward operation in the network.

        This method computes the output of the network for a given input by propagating the input through each layer. 
        In each layer, it calculates the weighted sum of the inputs and biases, and then applies a sigmoid activation function.  This process is repeated for all layers in the network.
        """
        activation = activation.reshape(-1, 1)
        for weight, bias in zip(weights, biases):
            # Calculate weighted sum for the current layer
            z = np.dot(weight, activation) + bias
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
            random.shuffle(training_data)

            # Iterate over each example in the training data
            for input_data, expected_output in training_data:
                # Perform backpropagation to compute the gradients of weights (grad_w) and biases (grad_b)
                grad_w, grad_b = self.backpropagation(
                    input_data, expected_output)
                # Update the weights of the network by subtracting the gradient scaled by the learning rate
                self.weights = [w - learning_rate *
                                gw for w, gw in zip(self.weights, grad_w)]
                # Update the biases in a similar way, adjusting them based on their gradients and the learning rate
                self.biases = [b - learning_rate *
                               gb for b, gb in zip(self.biases, grad_b)]
            if test_data:
                logging.debug(
                    f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                logging.debug(f"Epoch {epoch} finished")

    def backpropagation(self, input_data: np.ndarray, expected_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = input_data.reshape(-1, 1)
        # list to store all the activations, layer by layer
        activations = [input_data.reshape(-1, 1)]
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # logging.debug(f"\nActivations[-1]:\n{activations[-1]}")
        # backward pass
        # logging.debug(f"\nExpected output:\n{expected_output}")
        delta = -self.cost_derivative(activations[-1], expected_output) * \
            self.sigmoid_prime(zs[-1])
        grad_b[-1] = delta
        # logging.debug(
        #     f"\nDelta:\n{delta}"
        # )
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (grad_w, grad_b)

    def evaluate(self, test_data: np.ndarray) -> int:
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results = [(np.argmax(self.feedforward(x, self.weights, self.biases)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(x == np.argmax(y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Compute the gradient of the cost function with respect to the output layer activations."""

        return (output_activations - y)

    def sigmoid(self, z: np.ndarray) -> float:
        """Sigmoid activation function."""

        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z: np.ndarray) -> float:
        """Derivative of the sigmoid function."""

        return self.sigmoid(z) * (1 - self.sigmoid(z))
