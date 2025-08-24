"""
Main application for training and testing perceptron on logic gates
"""

import numpy as np
from neuron import Perceptron, plot_decision_boundary, plot_learning_curves
from datasets import get_all_gates, get_input_data, get_gate_data

def train_all_gates():
    """
    Train perceptron on all logic gates and return results
    
    Returns:
        dict: Training results for each gate
    """
    print("=" * 50)
    print("TRAINING PERCEPTRON ON LOGIC GATES")
    print("=" * 50)
    
    X = get_input_data()
    gates = get_all_gates()
    results = {}
    
    for gate_name, y in gates.items():
        print(f"\nTraining {gate_name} gate...")
        
        # Create and train perceptron
        perceptron = Perceptron(n_inputs=2, lr=0.1, epochs=50)
        perceptron.fit(X, y)
        
        # Make predictions and calculate accuracy
        predictions = perceptron.predict(X)
        accuracy = (predictions == y).mean()
        
        # Store results
        weights, bias = perceptron.get_weights()
        results[gate_name] = {
            "pred": predictions.tolist(),
            "acc": accuracy,
            "weights": weights,
            "bias": bias,
            "history": perceptron.history
        }
        
        print(f"{gate_name} training completed - Accuracy: {accuracy*100:.1f}%")
    
    return results

def display_results(results):
    """
    Display training results for all gates
    
    Args:
        results (dict): Training results from train_all_gates()
    """
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    
    gates = get_all_gates()
    
    for gate_name, info in results.items():
        print(f"\n{gate_name} GATE:")
        print(f"   Truth Table: {gates[gate_name].tolist()}")
        print(f"   Predicted:   {info['pred']}")
        print(f"   Accuracy:    {info['acc']*100:.1f}%")
        print(f"   Weights:     [{info['weights'][0]:.3f}, {info['weights'][1]:.3f}]")
        print(f"   Bias:        {info['bias']:.3f}")
        print(f"   Epochs:      {len(info['history'])}")

def visualize_results(results):
    """
    Create visualizations for the training results
    
    Args:
        results (dict): Training results from train_all_gates()
    """
    print("\n" + "=" * 40)
    print("GENERATING VISUALIZATIONS")
    print("=" * 40)
    
    # Plot learning curves
    print("\nDisplaying learning curves...")
    plot_learning_curves(results)
    
    # Plot decision boundaries
    print("\nDisplaying decision boundaries...")
    X = get_input_data()
    gates = get_all_gates()
    
    for gate_name, info in results.items():
        # Create perceptron with trained weights
        perceptron = Perceptron(n_inputs=2)
        perceptron.set_weights(info["weights"], info["bias"])
        
        # Plot decision boundary
        plot_decision_boundary(perceptron, X, gates[gate_name], 
                             f"{gate_name} Gate Decision Boundary")

def interactive_mode(results):
    """
    Interactive mode for testing individual inputs
    
    Args:
        results (dict): Training results from train_all_gates()
    """
    print("\n" + "=" * 50)
    print("INTERACTIVE TESTING MODE")
    print("=" * 50)
    print("Test the trained perceptrons with custom inputs!")
    print("Available gates: AND, OR, NAND, XOR")
    print("Type 'exit' to quit\n")
    
    while True:
        # Get gate choice
        gate_choice = input("Enter gate (AND/OR/NAND/XOR) or 'exit': ").upper().strip()
        
        if gate_choice == "EXIT":
            print("Goodbye!")
            break
            
        if gate_choice not in results:
            print(f"Invalid gate '{gate_choice}'! Available: {list(results.keys())}")
            continue
        
        try:
            # Get inputs
            x1 = int(input("   Input 1 (0 or 1): "))
            x2 = int(input("   Input 2 (0 or 1): "))
            
            if x1 not in [0, 1] or x2 not in [0, 1]:
                print("Inputs must be 0 or 1!")
                continue
                
        except ValueError:
            print("Invalid input! Please enter 0 or 1.")
            continue
        
        # Create perceptron with trained weights
        perceptron = Perceptron(n_inputs=2)
        perceptron.set_weights(results[gate_choice]["weights"], 
                              results[gate_choice]["bias"])
        
        # Make prediction
        output = perceptron.predict_single(np.array([x1, x2]))
        
        print(f"{gate_choice}({x1}, {x2}) = {output}")
        print("-" * 30)

def main():
    """
    Main function to run the complete perceptron training and testing pipeline
    """
    print("PERCEPTRON LOGIC GATE TRAINER")
    print("=" * 50)
    
    try:
        # Step 1: Train all gates
        results = train_all_gates()
        
        # Step 2: Display results
        display_results(results)
        
        # Step 3: Show visualizations
        visualize_results(results)
        
        # Step 4: Interactive testing
        interactive_mode(results)
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
