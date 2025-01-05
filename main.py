import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches


class NeuralNet:
    def __init__(self, input_size=32, hidden_size=16, output_size=3):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        hidden = self.sigmoid(np.dot(x, self.weights1))
        output = np.dot(hidden, self.weights2)
        return output

    def calculate_mse(self, inputs, targets):
        predictions = np.array([self.forward(x) for x in inputs])
        mse = np.mean((predictions - targets) ** 2)
        return mse


def load_data(file_path="spektar2.bin", data_type=np.int32, num_frequencies=32):
    """Load the spectral data from a binary file and reshape it into a 2D array.

    Args:
        file_path (str, optional): path to file. Defaults to "spektar2.bin".
        data_type (_type_, optional): Defaults to np.int32.
        num_frequencies (int, optional): Defaults to 32.

    Raises:
        ValueError: if the data size is not divisible by the number of frequency bins.

    Returns:
        list[np.int32]: Spectral data array.
    """
    data = np.fromfile(file_path, dtype=data_type)
    total_elements = data.size

    # Calculate the number of time steps based on the total elements and frequency bins
    if total_elements % num_frequencies != 0:
        raise ValueError(
            f"The data size ({total_elements}) is not divisible by the number of frequency bins ({num_frequencies})."
        )

    num_time_steps = total_elements // num_frequencies

    # Reshape the data into a 2D array (time vs frequency)
    data = data.reshape((num_time_steps, num_frequencies))

    # First half of the data is negative frequencies, so we need to shift it to the left
    half = num_frequencies // 2
    data = np.hstack((data[:, half:], data[:, :half]))

    return data


def animation(data, num_frequencies=32, fps=30, step=10000):
    """create an animation of the spectral data

    Args:
        data (list[np.int32]): Spectral data array.
        num_frequencies (int, optional): Defaults to 32.
        fps (int, optional): Defaults to 30.
        step (int, optional): Defaults to 10000.

    Returns:
        void: Display the animation.
    """
    # Calculate the interval between frames
    interval = 1000 / fps

    # Initialize the plot
    frequencies = np.linspace(1006, 1008, num_frequencies)
    fig, ax = plt.subplots(figsize=(8, 4))
    (line,) = ax.plot([], [], label="Row 0")
    ax.set_xlim(1006, 1008)
    ax.set_ylim(data.min(), data.max())
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Intensity")
    ax.set_title("Spectral Data Animation")
    ax.legend()
    ax.grid(True)

    # Update function for animation
    def update(frame):
        row_index = frame
        line.set_data(frequencies, data[row_index, :])
        line.set_label(f"Row {row_index}")
        ax.legend()
        return (line,)

    # Create animation
    num_frames = data.shape[0]
    frames = range(0, num_frames, step)
    ani = FuncAnimation(
        fig, update, frames=frames, blit=True, repeat=False, interval=interval
    )

    plt.show()


def plot(data):
    """Plot the spectral data as a heatmap

    Args:
        data (list[np.int32]): Spectral data array.
        num_frequencies (int, optional): Defaults to 32.

    Returns:
        void: Display the heatmap.
    """
    sample = 500
    data = data[::sample]

    num_time_steps = data.shape[0]
    plt.figure(figsize=(10, 6))
    plt.imshow(
        data, aspect="auto", cmap="turbo", extent=[1006, 1008, 0, num_time_steps]
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Samples")
    plt.title("Spectral Heatmap")
    plt.show()


def plot_clusters(data, cluster_labels):
    num_time_steps = data.shape[0]
    sample = 500
    data = data[::sample]
    cluster_labels = cluster_labels[::sample]

    plt.figure(figsize=(15, 6))

    # Plot the spectral heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(
        data, aspect="auto", cmap="turbo", extent=[1006, 1008, 0, num_time_steps]
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Samples")
    plt.title("Spectral Heatmap")

    # Prepare cluster_labels and plot them
    cluster_labels = np.expand_dims(cluster_labels, axis=1)

    plt.subplot(1, 2, 2)
    plt.imshow(
        cluster_labels,
        aspect="auto",
        cmap="tab20",
        extent=[0, 1, 0, num_time_steps],
    )
    plt.title("Cluster Labels")
    plt.legend(
        handles=[
            mpatches.Patch(
                color=plt.cm.tab20(i / cluster_labels.max()), label=f"Program {i}"
            )
            for i in range(4)
        ]
    )
    plt.yticks([])
    plt.xticks([])

    plt.tight_layout()
    plt.show()


def manual_clustering(data):
    size = data.shape[0]
    cluster_labels = np.zeros(size)
    cluster_labels[236500:350000] = 1
    cluster_labels[350000:1105000] = 2
    cluster_labels[1105000:1346000] = 3
    return cluster_labels[::-1]


def main():
    # File path to the binary file
    file_path = "spektar2.bin"
    num_frequencies = 32

    # Load the data
    data = load_data(
        file_path=file_path, data_type=np.int32, num_frequencies=num_frequencies
    )

    # Plot the data
    plot(data=data)

    # Manual clustering
    cluster_labels = manual_clustering(data)

    # Plot the data with cluster labels
    plot_clusters(data, cluster_labels)

    # Animate the data
    animation(data, num_frequencies=num_frequencies, fps=30, step=10000)


if __name__ == "__main__":
    main()
