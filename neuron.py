import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """Simple perceptron implementation for binary classification."""
    
    def __init__(self, n_inputs, lr=0.1, epochs=50):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.history = []

    def activation(self, z):
        return 1 if z >= 0 else 0

    def predict_single(self, x):
        """Predict output for a single input vector."""
        z = np.dot(self.w, x) + self.b
        return self.activation(z)

    def predict(self, X):
        """Predict outputs for multiple input vectors."""
        return np.array([self.predict_single(x) for x in X])

    def fit(self, X, y):
        """Train the perceptron using batch gradient descent."""
        self.history = []
        
        for epoch in range(self.epochs):
            errors = 0
            weight_update = np.zeros_like(self.w)
            bias_update = 0.0
            
            for xi, yi in zip(X, y):
                yhat = self.predict_single(xi)
                error = yi - yhat
                
                if error != 0:
                    weight_update += self.lr * error * xi
                    bias_update += self.lr * error
                    errors += 1
            
            self.w += weight_update
            self.b += bias_update
            self.history.append(errors)
            
            if errors == 0:
                print(f"Converged after {epoch + 1} epochs")
                break

    def get_weights(self):
        """Return current weights and bias."""
        return self.w.copy(), self.b

    def set_weights(self, weights, bias):
        """Set weights and bias to specific values."""
        self.w = weights.copy()
        self.b = bias

def plot_decision_boundary(perceptron, X, y, title):
    """Plot data points and decision boundary for visualization."""
    plt.figure(figsize=(6, 6))
    
    class_0_labeled = False
    class_1_labeled = False
    
    for i, point in enumerate(X):
        if y[i] == 0:
            label = "Class 0" if not class_0_labeled else ""
            plt.scatter(point[0], point[1], color="red", marker="o", s=100, label=label)
            class_0_labeled = True
        else:
            label = "Class 1" if not class_1_labeled else ""
            plt.scatter(point[0], point[1], color="blue", marker="x", s=100, label=label)
            class_1_labeled = True

    if not np.allclose(perceptron.w, 0):
        if perceptron.w[1] != 0:
            x_vals = np.linspace(-0.2, 1.2, 100)
            y_vals = -(perceptron.w[0] * x_vals + perceptron.b) / perceptron.w[1]
            plt.plot(x_vals, y_vals, "k--", linewidth=2, label="Decision Boundary")
        else:
            if perceptron.w[0] != 0:
                x_val = -perceptron.b / perceptron.w[0]
                plt.axvline(x_val, color="k", linestyle="--", linewidth=2, label="Decision Boundary")

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_learning_curves(results):
    """Display learning curves for different logic gates."""
    plt.figure(figsize=(12, 3))
    
    gates_to_plot = ["AND", "OR", "NAND", "XOR"]
    
    for i, gate in enumerate(gates_to_plot, start=1):
        if gate in results:
            plt.subplot(1, 4, i)
            epochs = range(1, len(results[gate]["history"]) + 1)
            plt.plot(epochs, results[gate]["history"], 'b-', linewidth=2)
            plt.title(f"{gate} Gate Learning")
            plt.xlabel("Epoch")
            plt.ylabel("Errors")
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
