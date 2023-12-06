import numpy as np
import random
from typing import List, Tuple


class Network():

    def __init__(self, layers: List[int]):
        self.layers_count = len(layers)
        self.layers = layers
        # Initialize weight matrices for each layer (expect the input layer)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]
        # Initialize bias matrices for each layer (except the input layer)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def feedforward(self, activation: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> float:
        """
        Perform the feedforward operation in the network.

        This method computes the output of the network for a given input by propagating the input through each layer. 
        In each layer, it calculates the weighted sum of the inputs and biases, and then applies a sigmoid activation function.  This process is repeated for all layers in the network.
        """

        for w, b in zip(weights, biases):
            z = np.dot(w, activation) + b
            activation = self.sigmoid(z)
        return activation

    def stochastic_gradient_descent(
        self,
        training_data: Tuple[np.ndarray, np.ndarray],
        learning_rate: float,
        epochs: int
    ) -> None:
        """
        Perform Stochastic Gradient Descent (SGD) to optimize the network.

        This function updates the weights and biases of the network based on the gradients computed
        from each training sample or a small batch of samples.
        """

        # Loop over each epoch (an epoch is one complete pass through the entire training set)
        for _ in range(epochs):
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

    def backpropagation(self, input_data: np.ndarray, expected_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        
        pass

    def cost_derivative(self, output_activations, y):
        """Compute the gradient of the cost function with respect to the output layer activations."""

        return (output_activations - y)

    def sigmoid(self, z: np.ndarray) -> float:
        """Sigmoid activation function"""

        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z: np.ndarray) -> float:
        """Derivative of the sigmoid function."""

        return self.sigmoid(z) * (1 - self.sigmoid(z))
