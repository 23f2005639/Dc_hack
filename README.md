# Perceptron Logic Gate Trainer

## Theory of Neural Networks and Perceptrons

### Mathematical Foundation

A perceptron is the simplest form of an artificial neural network, inspired by biological neurons. It functions as a linear binary classifier that can learn to separate data into two classes.

### Core Mathematical Concepts

#### 1. Linear Combination
The perceptron computes a weighted sum of its inputs:
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```
Where:
- `w₁, w₂, ..., wₙ` are the weights
- `x₁, x₂, ..., xₙ` are the input features
- `b` is the bias term
- `z` is the linear combination

In vector notation: `z = w · x + b`

#### 2. Activation Function
The perceptron uses a step activation function:
```
f(z) = 1 if z ≥ 0
       0 if z < 0
```

This creates a decision boundary where the perceptron outputs 1 for one class and 0 for the other.

#### 3. Learning Algorithm
The perceptron learns through the perceptron learning rule:

For each training example (x, y):
1. Calculate prediction: `ŷ = f(w · x + b)`
2. Calculate error: `error = y - ŷ`
3. Update weights: `w = w + η × error × x`
4. Update bias: `b = b + η × error`

Where `η` (eta) is the learning rate.

#### 4. Decision Boundary
The decision boundary is a hyperplane defined by:
```
w₁x₁ + w₂x₂ + b = 0
```

For 2D inputs, this creates a line that separates the two classes. Points on one side are classified as 1, points on the other side as 0.

#### 5. Convergence Theorem
The perceptron convergence theorem states that if the data is linearly separable, the perceptron learning algorithm will find a solution in finite steps. However, if the data is not linearly separable (like XOR), the perceptron will never converge.

### Limitations
- Can only solve linearly separable problems
- XOR problem cannot be solved with a single perceptron
- Limited to binary classification

## Key Features

### 🎯 Multiple Interfaces
- **Command Line**: Quick batch training and testing
- **Web Interface**: Interactive Streamlit dashboard
- **Animation**: Real-time PyGame visualization
- **Jupyter Notebook**: Educational exploration and experimentation

### 📊 Comprehensive Visualizations
- **Decision Boundaries**: See how the perceptron separates data
- **Learning Curves**: Track training progress over epochs
- **Real-time Animation**: Watch weights and boundaries evolve
- **Network Diagrams**: Visual representation of the neural network

### 🎮 Interactive Learning
- **Live Training**: Watch the perceptron learn step-by-step
- **Parameter Tuning**: Adjust learning rates and epochs
- **Gate Switching**: Compare different logic gate behaviors
- **Pause/Resume**: Control training pace for better understanding

### 📚 Educational Content
- **Mathematical Theory**: Detailed explanations of perceptron concepts
- **Visual Learning**: Multiple visualization approaches
- **Hands-on Experience**: Interactive experimentation
- **XOR Problem**: Demonstrates perceptron limitations

## Project Structure

### Core Files

#### `neuron.py`
Contains the main Perceptron class implementation with the following key components:
- **Perceptron class**: Implements the mathematical model described above
- **Training method**: Uses the perceptron learning rule for weight updates
- **Prediction methods**: For single inputs and batch predictions
- **Visualization functions**: Creates decision boundary plots and learning curves
- **Weight management**: Methods to get and set weights for experimentation

#### `datasets.py`
Provides training data for logic gates:
- **Input data**: 2D truth table inputs `[[0,0], [0,1], [1,0], [1,1]]`
- **Logic gates**: AND, OR, NAND, and XOR truth tables
- **Data access functions**: Clean interface to retrieve gate-specific datasets
- **Validation**: Ensures proper gate names and data consistency

#### `main.py`
Command-line interface for training and testing:
- **Batch training**: Trains perceptrons on all logic gates automatically
- **Results display**: Shows accuracy, weights, and convergence information
- **Visualization**: Generates plots for decision boundaries and learning curves
- **Interactive mode**: Allows manual testing with custom inputs
- **Comprehensive reporting**: Detailed analysis of training results

#### `streamlit_app.py`
Modern web-based user interface:
- **Interactive training**: Real-time parameter adjustment and training
- **Visual dashboard**: Beautiful plots and metrics display
- **Parameter tuning**: Sliders for learning rate and epoch adjustment
- **Live testing**: Interactive prediction interface
- **Responsive design**: Works on different screen sizes
- **Educational content**: Built-in explanations and help text

#### `run_app.py`
Simple launcher script:
- **One-click startup**: Launches the Streamlit application
- **Configuration**: Sets proper server settings
- **Error handling**: Provides helpful error messages
- **Cross-platform**: Works on different operating systems

#### `two_step_perceptron_for_XOR.ipynb`
Jupyter notebook demonstrating advanced concepts:
- **XOR problem**: Shows why single perceptron fails on XOR
- **Multi-layer solution**: Implements two-layer perceptron for XOR
- **Step-by-step explanation**: Educational walkthrough of the solution
- **Visualizations**: Interactive plots showing the learning process
- **Experimentation**: Code cells for trying different approaches

#### `pygame_training_animation.py`
Real-time animated visualization of neural network training:
- **Live training animation**: Watch the perceptron learn in real-time
- **Interactive controls**: Pause/resume, restart, and switch between logic gates
- **Visual decision boundary**: See how the decision line evolves during training
- **Network diagram**: Real-time display of weights and bias values
- **Error tracking**: Live graph showing training errors over epochs
- **Multiple logic gates**: Visualize AND, OR, NAND, and XOR gate training
- **Educational interface**: Clear legends and status indicators
- **Smooth animations**: Optimized for learning and demonstration purposes

## How to Run

### Prerequisites
Make sure you have Python 3.7 or higher installed with the following packages:
```bash
pip install numpy matplotlib streamlit plotly pandas pygame
```

### Running Options

#### 1. Command Line Interface
```bash
python main.py
```

#### 2. Web-based Interface
```bash
python run_app.py
```
This launches the Streamlit web application for an interactive experience.

#### 3. Real-time Training Animation
```bash
python pygame_training_animation.py
```
This opens a PyGame window with animated neural network training visualization.

**Animation Controls:**
- `SPACE` - Pause/Resume training
- `R` - Restart current training
- `1` - Switch to AND gate
- `2` - Switch to OR gate  
- `3` - Switch to NAND gate
- `4` - Switch to XOR gate

#### 4. Jupyter Notebook
Open `two_step_perceptron_for_XOR.ipynb` in Jupyter Notebook or JupyterLab for interactive exploration.
