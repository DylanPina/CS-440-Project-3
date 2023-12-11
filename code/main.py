from config import init_logging
from wire_diagram import WireDiagramDataLoader
from logistic_regression import LogisticRegression

NON_LINEAR_INPUT_LAYER_SIZE = 2360
LINEAR_INPUT_LAYER_SIZE = 1600

TRAINING_DATA_SIZE = 5000
VALIDATION_DATA_SIZE = 0
TESTING_DATA_SIZE = 1000
LEARNING_RATE = 0.01
EPOCHS = 100

if __name__ == '__main__':
    init_logging()

    training_data, validation_data, testing_data = WireDiagramDataLoader(
        TRAINING_DATA_SIZE, VALIDATION_DATA_SIZE, TESTING_DATA_SIZE).load_data(non_linear_features=False)

    t1_linear = LogisticRegression(
        input_layer_size=LINEAR_INPUT_LAYER_SIZE,
        training_data=training_data,
        testing_data=testing_data,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS)

    t1_linear.stochastic_gradient_descent()
    t1_linear.plot_loss(linear=True)
    t1_linear.plot_success(linear=True)

    training_data, validation_data, testing_data = WireDiagramDataLoader(
        TRAINING_DATA_SIZE, VALIDATION_DATA_SIZE, TESTING_DATA_SIZE).load_data(non_linear_features=True)

    t1_non_linear = LogisticRegression(
        input_layer_size=NON_LINEAR_INPUT_LAYER_SIZE,
        training_data=training_data,
        testing_data=testing_data,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS)

    t1_non_linear.stochastic_gradient_descent()
    t1_non_linear.plot_loss(linear=False)
    t1_non_linear.plot_success(linear=False)
