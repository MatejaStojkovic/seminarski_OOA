import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
        print(frame)
        row_index = frame
        line.set_data(frequencies, data[row_index, :])
        line.set_label(f"Row {row_index}")
        ax.legend()
        return (line,)

    # Create animation
    num_frames = data.shape[0]  # Number of frames (one per row)
    frames = range(0, num_frames, step)
    print(f"Number of frames: {num_frames}")
    ani = FuncAnimation(
        fig, update, frames=frames, blit=True, repeat=False, interval=interval
    )

    # Show the animation
    plt.show()


def plot(data):
    """Plot the spectral data as a heatmap

    Args:
        data (list[np.int32]): Spectral data array.
        num_frequencies (int, optional): Defaults to 32.

    Returns:
        void: Display the heatmap.
    """
    num_time_steps = data.shape[0]
    plt.figure(figsize=(10, 6))
    plt.imshow(
        data, aspect="auto", cmap="turbo", extent=[1006, 1008, 0, num_time_steps]
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (s)")
    plt.title("Spectral Heatmap")
    plt.show()


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

    # Animate the data
    animation(data, num_frequencies=num_frequencies, fps=30, step=10000)


if __name__ == "__main__":
    main()
