import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import time
from datetime import datetime
import pickle
import os


class NeuralNet:
    def __init__(self, input_size=32, hidden_size=16, output_size=4, learning_rate=0.01):
        # Initialize weights with better scaling
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.learning_rate = learning_rate
        
        # Store intermediate values for backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def sigmoid(self, x):
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward(self, x):
        # Forward pass with stored intermediate values
        self.z1 = np.dot(x, self.weights1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2)
        self.a2 = self.z2  # Linear output for regression
        return self.a2
    
    def backward(self, x, y_true, y_pred):
        m = x.shape[0] if len(x.shape) > 1 else 1
        
        # Output layer gradients
        dz2 = y_pred - y_true
        dw2 = np.dot(self.a1.T, dz2) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self.sigmoid_derivative(self.z1)
        dw1 = np.dot(x.T, dz1) / m
        
        # Update weights
        self.weights2 -= self.learning_rate * dw2
        self.weights1 -= self.learning_rate * dw1
    
    def train_step(self, x, y):
        # Single training step
        y_pred = self.forward(x)
        self.backward(x, y, y_pred)
        loss = np.mean((y_pred - y) ** 2)
        return loss
    
    def train(self, X_train, y_train, epochs=1000, batch_size=32, verbose=True):
        losses = []
        start_time = time.time()
        
        print(f"Training started at: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_losses = []
            num_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size != 0 else 0)
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs}:")
            
            for batch_idx, i in enumerate(range(0, len(X_train), batch_size)):
                batch_start_time = time.time()
                
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                if len(batch_X) > 0:
                    loss = self.train_step(batch_X, batch_y)
                    epoch_losses.append(loss)
                
                # Show batch progress within epoch
                if verbose:
                    batch_progress = (batch_idx + 1) / num_batches
                    bar_length = 40
                    filled_length = int(bar_length * batch_progress)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    
                    # Calculate time remaining for this epoch
                    batch_time = time.time() - batch_start_time
                    elapsed_epoch_time = time.time() - epoch_start_time
                    if batch_idx > 0:
                        avg_batch_time = elapsed_epoch_time / (batch_idx + 1)
                        remaining_batches = num_batches - (batch_idx + 1)
                        eta_epoch = remaining_batches * avg_batch_time
                    else:
                        eta_epoch = 0
                    
                    def format_time(seconds):
                        if seconds < 60:
                            return f"{seconds:.1f}s"
                        elif seconds < 3600:
                            return f"{int(seconds//60)}m {int(seconds%60)}s"
                        else:
                            return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"
                    
                    # Update progress line (overwrite same line)
                    print(f"\rBatch {batch_idx+1:4d}/{num_batches} |{bar}| {batch_progress*100:5.1f}% | "
                          f"Loss: {loss:.6f} | ETA: {format_time(eta_epoch)}", end='', flush=True)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            epoch_time = time.time() - epoch_start_time
            
            if verbose:
                print(f"\nEpoch {epoch+1} completed - Avg Loss: {avg_loss:.6f} - Time: {format_time(epoch_time)}")
                
                # Show overall training progress
                overall_progress = (epoch + 1) / epochs
                elapsed_time = time.time() - start_time
                if epoch > 0:
                    avg_epoch_time = elapsed_time / (epoch + 1)
                    remaining_epochs = epochs - (epoch + 1)
                    eta_total = remaining_epochs * avg_epoch_time
                    print(f"Overall Progress: {overall_progress*100:.1f}% | ETA Total: {format_time(eta_total)}")
        
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training completed in {format_time(total_time)}")
        print(f"Final loss: {losses[-1]:.6f}")
        
        return losses
    
    def calculate_mse(self, inputs, targets):
        predictions = np.array([self.forward(x) for x in inputs])
        mse = np.mean((predictions - targets) ** 2)
        return mse
    
    def predict(self, X):
        predictions = []
        for x in X:
            pred = self.forward(x)
            predictions.append(pred)
        return np.array(predictions)
    
    def save_model(self, filepath):
        """Save the trained model to a file.
        
        Args:
            filepath: Path where to save the model
        """
        model_data = {
            'weights1': self.weights1,
            'weights2': self.weights2,
            'learning_rate': self.learning_rate,
            'input_size': self.weights1.shape[0],
            'hidden_size': self.weights1.shape[1],
            'output_size': self.weights2.shape[1]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            NeuralNet: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance with same architecture
        model = cls(
            input_size=model_data['input_size'],
            hidden_size=model_data['hidden_size'],
            output_size=model_data['output_size'],
            learning_rate=model_data['learning_rate']
        )
        
        # Load the trained weights
        model.weights1 = model_data['weights1']
        model.weights2 = model_data['weights2']
        
        print(f"Model loaded from: {filepath}")
        return model


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


def prepare_data_for_training(data, cluster_labels, train_ratio=0.8):
    """Prepare spectral data for neural network training.
    
    Args:
        data: Spectral data array (num_samples, num_frequencies)
        cluster_labels: Cluster labels for each sample
        train_ratio: Ratio of data to use for training
    
    Returns:
        X_train, X_test, y_train, y_test, data_stats: Training and testing data plus normalization stats
    """
    # Calculate normalization statistics
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_stats = {'mean': data_mean, 'std': data_std}
    
    # Normalize the input data
    X = (data - data_mean) / (data_std + 1e-8)
    
    # Convert cluster labels to one-hot encoding
    num_classes = int(np.max(cluster_labels)) + 1
    y = np.zeros((len(cluster_labels), num_classes))
    for i, label in enumerate(cluster_labels):
        y[i, int(label)] = 1
    
    # Split data into train and test sets
    split_idx = int(len(X) * train_ratio)
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test, data_stats


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data.
    
    Args:
        model: Trained neural network
        X_test: Test input data
        y_test: Test target data
    
    Returns:
        accuracy, predictions
    """
    predictions = model.predict(X_test)
    
    # Convert predictions and true labels to class indices
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(pred_classes == true_classes)
    
    return accuracy, pred_classes


def plot_training_history(losses):
    """Plot the training loss history.
    
    Args:
        losses: List of loss values during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()


def train_neural_network(data, cluster_labels, hidden_size=64, learning_rate=0.001, epochs=200):
    """Train a neural network on the spectral data.
    
    Args:
        data: Spectral data array
        cluster_labels: Cluster labels
        hidden_size: Size of hidden layer
        learning_rate: Learning rate for training
        epochs: Number of training epochs
    
    Returns:
        Trained model, training history, test accuracy, data_stats
    """
    X_train, X_test, y_train, y_test, data_stats = prepare_data_for_training(data, cluster_labels)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    model = NeuralNet(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate
    )
    
    print(f"\nTraining neural network...")
    print(f"Architecture: {input_size} -> {hidden_size} -> {output_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    
    losses = model.train(X_train, y_train, epochs=epochs, batch_size=64, verbose=True)
    
    print("\nEvaluating model...")
    accuracy, predictions = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Plot training history
    plot_training_history(losses)
    
    return model, losses, accuracy, data_stats


def save_model_with_preprocessing(model, data_stats, model_path="trained_model.pkl"):
    """Save model along with preprocessing statistics.
    
    Args:
        model: Trained neural network
        data_stats: Dictionary with mean and std for normalization
        model_path: Path to save the model
    """
    model_package = {
        'model': model,
        'data_mean': data_stats['mean'],
        'data_std': data_stats['std'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"Model and preprocessing stats saved to: {model_path}")
    return model_path


def load_model_with_preprocessing(model_path="trained_model.pkl"):
    """Load model along with preprocessing statistics.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        tuple: (model, data_mean, data_std)
    """
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    print(f"Model loaded from: {model_path}")
    print(f"Model was trained on: {model_package['timestamp']}")
    
    return model_package['model'], model_package['data_mean'], model_package['data_std']


def predict_single_sample(model, data_mean, data_std, sample_input):
    """Make prediction on a single sample.
    
    Args:
        model: Trained neural network
        data_mean: Mean values for normalization
        data_std: Standard deviation values for normalization
        sample_input: Input sample (32 frequency values)
        
    Returns:
        tuple: (predicted_class, confidence_scores)
    """
    # Normalize the input
    normalized_input = (sample_input - data_mean) / (data_std + 1e-8)
    
    # Make prediction
    prediction = model.forward(normalized_input.reshape(1, -1))
    
    # Apply softmax to get probabilities
    exp_pred = np.exp(prediction - np.max(prediction))
    probabilities = exp_pred / np.sum(exp_pred)
    
    predicted_class = np.argmax(probabilities)
    
    return predicted_class, probabilities.flatten()


def interactive_prediction_interface(model_path="trained_model.pkl"):
    """Interactive interface for making predictions.
    
    Args:
        model_path: Path to the saved model
    """
    try:
        model, data_mean, data_std = load_model_with_preprocessing(model_path)
        print("\n" + "="*50)
        print("INTERACTIVE PREDICTION INTERFACE")
        print("="*50)
        print("Enter 32 frequency values (space-separated) or 'quit' to exit.")
        print("Classes: 0=Background, 1=Program1, 2=Program2, 3=Program3\n")
        
        while True:
            try:
                user_input = input("Enter 32 frequency values: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                # Parse input
                values = list(map(float, user_input.split()))
                
                if len(values) != 32:
                    print(f"Error: Expected 32 values, got {len(values)}. Please try again.")
                    continue
                
                # Make prediction
                sample_input = np.array(values)
                predicted_class, probabilities = predict_single_sample(
                    model, data_mean, data_std, sample_input
                )
                
                # Display results
                print(f"\nPrediction Results:")
                print(f"Predicted Class: {predicted_class}")
                print(f"Confidence Scores:")
                for i, prob in enumerate(probabilities):
                    print(f"  Class {i}: {prob:.4f} ({prob*100:.2f}%)")
                print("-" * 30)
                
            except ValueError:
                print("Error: Please enter valid numbers separated by spaces.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
                
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please train a model first.")
    except Exception as e:
        print(f"Error loading model: {e}")


def test_with_random_samples(model_path="trained_model.pkl", num_samples=5):
    """Test the model with random samples from the dataset.
    
    Args:
        model_path: Path to the saved model
        num_samples: Number of random samples to test
    """
    try:
        # Load model
        model, data_mean, data_std = load_model_with_preprocessing(model_path)
        
        # Load original data for testing
        data = load_data("spektar2.bin")
        cluster_labels = manual_clustering(data)
        
        print("\n" + "="*50)
        print("TESTING WITH RANDOM SAMPLES")
        print("="*50)
        
        # Test with random samples
        random_indices = np.random.choice(len(data), num_samples, replace=False)
        
        for i, idx in enumerate(random_indices):
            sample = data[idx]
            true_label = int(cluster_labels[idx])
            
            predicted_class, probabilities = predict_single_sample(
                model, data_mean, data_std, sample
            )
            
            print(f"\nSample {i+1} (Index {idx}):")
            print(f"True Label: {true_label}")
            print(f"Predicted: {predicted_class}")
            print(f"Correct: {'✓' if predicted_class == true_label else '✗'}")
            print(f"Confidence: {probabilities[predicted_class]:.4f}")
            
    except Exception as e:
        print(f"Error during testing: {e}")


def main():
    file_path = "spektar2.bin"
    num_frequencies = 32
    model_name = input("Enter the model name (default is 'trained_model.pkl'): ").strip()
    if not model_name:
        model_name = "trained_model.pkl"
    model_path = model_name

    if os.path.exists(model_path):
        choice = input(f"\nFound existing model at {model_path}. Do you want to:\n1. Load existing model\n2. Train new model\nEnter choice (1/2): ").strip()
        
        if choice == '1':
            print("Loading existing model...")
        elif choice == '2':
            print("Training new model...")
        else:
            print("Invalid choice. Defaulting to training new model.")
    else:
        print("No existing model found. Training new model...")

    # Load data
    data = load_data(
        file_path=file_path, data_type=np.int32, num_frequencies=num_frequencies
    )
    print(f"Loaded data shape: {data.shape}")
    plot(data=data)

    cluster_labels = manual_clustering(data)
    print(f"Cluster labels shape: {cluster_labels.shape}")
    print(f"Unique clusters: {np.unique(cluster_labels)}")
    plot_clusters(data, cluster_labels)
    
    if choice != '1':
        # Train new model
        model, losses, accuracy, data_stats = train_neural_network(
            data=data,
            cluster_labels=cluster_labels,
            hidden_size=32,
            learning_rate=0.001,
            epochs=200
        )
        
        print(f"\nFinal training accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        model_path = save_model_with_preprocessing(model, data_stats, model_path)
    else:
        print("\nLoading model for predictions...")

    # Test the model with random samples
    test_with_random_samples(model_path, num_samples=3)
    
    # Ask user if they want to use interactive interface
    print("\n" + "="*50)
    print("INTERACTIVE PREDICTION MODE")
    print("="*50)
    
    while True:
        choice = input("\nWould you like to:\n1. Enter custom values for prediction\n2. Test with more random samples\n3. Exit\nEnter choice (1/2/3): ").strip()
        
        if choice == '1':
            interactive_prediction_interface(model_path)
        elif choice == '2':
            num_samples = int(input("How many random samples to test? "))
            test_with_random_samples(model_path, num_samples)
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Optionally animate the data (comment out if not needed)
    # animation(data, num_frequencies=num_frequencies, fps=30, step=10000)


if __name__ == "__main__":
    main()
