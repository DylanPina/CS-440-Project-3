from config import init_logging
from wire_diagram import WireDiagramDataLoader
from logistic_regression import LogisticRegression

if __name__ == '__main__':
    init_logging()

    training_data, validation_data, testing_data = WireDiagramDataLoader(
        5000, 0, 5000).load_data()

    logRegression = LogisticRegression(
        input_layer_size=1600,
        training_data=training_data,
        testing_data=testing_data,
        learning_rate=0.01,
        epochs=1000)
    
    logRegression.stochastic_gradient_descent()
    logRegression.plot_loss()
    logRegression.plot_success()