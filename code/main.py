import logging
import numpy as np
from softmax_regression import SoftmaxRegression
from config import init_logging
from wire_diagram import WireDiagramDataLoader
from logistic_regression import LogisticRegression

LINEAR_INPUT_LAYER_SIZE = 1600
NON_LINEAR_INPUT_LAYER_SIZE = 3044

TRAINING_DATA_SIZE = 5000
VALIDATION_DATA_SIZE = 0
TESTING_DATA_SIZE = 1000
LEARNING_RATE = 0.01
EPOCHS = 100
LAMBDA_TERM = 0.1
PATIENCE = 20

if __name__ == '__main__':
    init_logging()

    training_data, validation_data, testing_data = WireDiagramDataLoader(
        TRAINING_DATA_SIZE, VALIDATION_DATA_SIZE, TESTING_DATA_SIZE).load_safety_data(non_linear_features=False)

    t1_linear = LogisticRegression(
        input_layer_size=LINEAR_INPUT_LAYER_SIZE,
        training_data=training_data,
        testing_data=testing_data,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        lambda_term=LAMBDA_TERM,
        patience=PATIENCE)

    t1_linear.stochastic_gradient_descent()

    training_data, validation_data, testing_data = WireDiagramDataLoader(
        TRAINING_DATA_SIZE, VALIDATION_DATA_SIZE, TESTING_DATA_SIZE).load_safety_data(non_linear_features=True)

    t1_non_linear = LogisticRegression(
        input_layer_size=NON_LINEAR_INPUT_LAYER_SIZE,
        training_data=training_data,
        testing_data=testing_data,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        lambda_term=LAMBDA_TERM,
        patience=PATIENCE)

    t1_non_linear.stochastic_gradient_descent()

    training_data, validation_data, testing_data = WireDiagramDataLoader(
        TRAINING_DATA_SIZE, VALIDATION_DATA_SIZE, TESTING_DATA_SIZE).load_cut_data(non_linear_features=False)

    t2_linear = SoftmaxRegression(
        input_layer_size=LINEAR_INPUT_LAYER_SIZE,
        training_data=training_data,
        testing_data=testing_data,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        lambda_term=LAMBDA_TERM,
        patience=PATIENCE)

    t2_linear.stochastic_gradient_descent()

    training_data, validation_data, testing_data = WireDiagramDataLoader(
        TRAINING_DATA_SIZE, VALIDATION_DATA_SIZE, TESTING_DATA_SIZE).load_cut_data(non_linear_features=True)

    t2_non_linear = SoftmaxRegression(
        input_layer_size=NON_LINEAR_INPUT_LAYER_SIZE,
        training_data=training_data,
        testing_data=testing_data,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        lambda_term=LAMBDA_TERM,
        patience=PATIENCE)

    t2_non_linear.stochastic_gradient_descent()
