"""
Dataset module for logic gates
Contains training data and target values for various logic gates
"""

import numpy as np

# Training data for logic gates (truth table inputs)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

# Target outputs for different logic gates
gates = {
    "AND":  np.array([0, 0, 0, 1]),
    "OR":   np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "XOR":  np.array([0, 1, 1, 0])
}

def get_gate_data(gate_name):
    
    if gate_name.upper() not in gates:
        raise ValueError(f"Unknown gate: {gate_name}. Available gates: {list(gates.keys())}")
    
    return X, gates[gate_name.upper()]

def get_all_gates():
    
    return gates.copy()

def get_input_data():
    
    return X.copy()
