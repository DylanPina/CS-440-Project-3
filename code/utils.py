from typing import List
import matplotlib.pyplot as plt


def plot(file_name: str, title: str, x_name: str, y_name: str, data: List[int]) -> None:
    """Plot some data and save it to a file."""

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(data)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    # Save the figure
    fig.savefig(f"./graphs/{file_name}.png")


def evaluate_activation(activation: float) -> int:
    return 1 if activation >= 0.5 else 0
