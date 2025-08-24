"""
Streamlit Web Application for Perceptron Logic Gate Trainer
A beautiful interactive UI for training and testing perceptrons on logic gates
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Import our custom modules
from neuron import Perceptron
from datasets import get_all_gates, get_input_data

# Configure page
st.set_page_config(
    page_title="Perceptron Logic Gate Trainer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e7d32;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def create_decision_boundary_plot(perceptron, X, y, title):
    """Create an interactive decision boundary plot using Plotly"""
    fig = go.Figure()
    
    # Add data points
    colors = ['red' if label == 0 else 'blue' for label in y]
    symbols = ['circle' if label == 0 else 'x' for label in y]
    
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(
            color=colors,
            size=15,
            symbol=symbols,
            line=dict(width=2, color='black')
        ),
        text=[f'({x[0]}, {x[1]}) → {y[i]}' for i, x in enumerate(X)],
        hovertemplate='Input: %{text}<br>Class: %{marker.color}<extra></extra>',
        name='Data Points'
    ))
    
    # Add decision boundary if weights are not zero
    if not np.allclose(perceptron.w, 0):
        if perceptron.w[1] != 0:
            x_vals = np.linspace(-0.2, 1.2, 100)
            y_vals = -(perceptron.w[0] * x_vals + perceptron.b) / perceptron.w[1]
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                line=dict(color='green', width=3, dash='dash'),
                name='Decision Boundary',
                hovertemplate='Decision Boundary<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color='#1f77b4')),
        xaxis_title="Input 1",
        yaxis_title="Input 2",
        xaxis=dict(range=[-0.2, 1.2], gridcolor='lightgray'),
        yaxis=dict(range=[-0.2, 1.2], gridcolor='lightgray'),
        plot_bgcolor='white',
        showlegend=True,
        width=400,
        height=400
    )
    
    return fig

def create_learning_curve_plot(history, gate_name):
    """Create a learning curve plot"""
    if not history:
        return None
        
    fig = go.Figure()
    
    epochs = list(range(1, len(history) + 1))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history,
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        name='Training Errors',
        hovertemplate='Epoch: %{x}<br>Errors: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f'{gate_name} Gate Learning Curve', x=0.5, font=dict(size=16)),
        xaxis_title="Epoch",
        yaxis_title="Number of Errors",
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        width=400,
        height=300
    )
    
    return fig

def train_perceptron(gate_name, X, y, lr=0.1, epochs=50):
    """Train a perceptron for a specific gate"""
    perceptron = Perceptron(n_inputs=2, lr=lr, epochs=epochs)
    perceptron.fit(X, y)
    
    predictions = perceptron.predict(X)
    accuracy = (predictions == y).mean()
    weights, bias = perceptron.get_weights()
    
    return {
        'perceptron': perceptron,
        'predictions': predictions,
        'accuracy': accuracy,
        'weights': weights,
        'bias': bias,
        'history': perceptron.history
    }

def main():
    # Header
    st.markdown('<div class="main-header">Perceptron Logic Gate Trainer</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Training parameters
        st.markdown("#### Training Parameters")
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        max_epochs = st.slider("Max Epochs", 10, 200, 50, 10)
        
        # Gate selection
        st.markdown("#### Select Logic Gates")
        gates = get_all_gates()
        selected_gates = st.multiselect(
            "Choose gates to train:",
            list(gates.keys()),
            default=list(gates.keys())
        )
        
        # Auto-train toggle
        auto_train = st.checkbox("Auto-train on parameter change", value=True)
        
        # Manual train button
        train_button = st.button("🚀 Train Perceptrons", type="primary")
        
        # Information box
        st.markdown("""
        <div class="info-box">
        <strong> How it works:</strong><br>
        • Perceptron learns to classify logic gate inputs<br>
        • Adjusts weights based on prediction errors<br>
        • Converges when all predictions are correct<br>
        • XOR gate is not linearly separable!
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if not selected_gates:
        st.warning("Please select at least one logic gate from the sidebar!")
        return
    
    # Initialize session state
    if 'trained_results' not in st.session_state:
        st.session_state.trained_results = {}
    
    # Training trigger
    should_train = train_button or (auto_train and (
        'last_lr' not in st.session_state or 
        st.session_state.last_lr != learning_rate or
        'last_epochs' not in st.session_state or
        st.session_state.last_epochs != max_epochs or
        'last_gates' not in st.session_state or
        st.session_state.last_gates != selected_gates
    ))
    
    if should_train:
        # Store current parameters
        st.session_state.last_lr = learning_rate
        st.session_state.last_epochs = max_epochs
        st.session_state.last_gates = selected_gates
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        X = get_input_data()
        st.session_state.trained_results = {}
        
        for i, gate_name in enumerate(selected_gates):
            status_text.text(f"Training {gate_name} gate...")
            
            y = gates[gate_name]
            result = train_perceptron(gate_name, X, y, learning_rate, max_epochs)
            st.session_state.trained_results[gate_name] = result
            
            progress_bar.progress((i + 1) / len(selected_gates))
        
        status_text.text("Training completed! 🎉")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
    
    # Display results if available
    if st.session_state.trained_results:
        # Summary metrics
        st.markdown('<div class="sub-header"> Training Results Summary</div>', unsafe_allow_html=True)
        
        # Create metrics columns
        cols = st.columns(len(selected_gates))
        for i, gate_name in enumerate(selected_gates):
            if gate_name in st.session_state.trained_results:
                result = st.session_state.trained_results[gate_name]
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{gate_name}</h3>
                        <h2>{result['accuracy']*100:.1f}%</h2>
                        <p>Accuracy</p>
                        <small>{len(result['history'])} epochs</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed results
        st.markdown('<div class="sub-header"> Detailed Analysis</div>', unsafe_allow_html=True)
        
        # Truth tables and predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Truth Tables & Predictions")
            X = get_input_data()
            
            for gate_name in selected_gates:
                if gate_name in st.session_state.trained_results:
                    result = st.session_state.trained_results[gate_name]
                    
                    df = pd.DataFrame({
                        'Input 1': X[:, 0].astype(int),
                        'Input 2': X[:, 1].astype(int),
                        f'{gate_name} Expected': gates[gate_name],
                        f'{gate_name} Predicted': result['predictions'],
                        'Correct': gates[gate_name] == result['predictions']
                    })
                    
                    st.markdown(f"**{gate_name} Gate**")
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
        
        with col2:
            st.markdown("#### ⚖️ Model Parameters")
            
            for gate_name in selected_gates:
                if gate_name in st.session_state.trained_results:
                    result = st.session_state.trained_results[gate_name]
                    
                    st.markdown(f"**{gate_name} Gate Parameters:**")
                    st.write(f"• Weight 1: {result['weights'][0]:.3f}")
                    st.write(f"• Weight 2: {result['weights'][1]:.3f}")
                    st.write(f"• Bias: {result['bias']:.3f}")
                    st.write(f"• Final Accuracy: {result['accuracy']*100:.1f}%")
                    st.write(f"• Epochs to Converge: {len(result['history'])}")
                    st.markdown("---")
        
        # Visualizations
        st.markdown('<div class="sub-header">Visualizations</div>', unsafe_allow_html=True)
        
        # Decision boundaries
        st.markdown("####  Decision Boundaries")
        viz_cols = st.columns(2)
        
        for i, gate_name in enumerate(selected_gates):
            if gate_name in st.session_state.trained_results:
                result = st.session_state.trained_results[gate_name]
                X = get_input_data()
                y = gates[gate_name]
                
                fig = create_decision_boundary_plot(
                    result['perceptron'], X, y, f"{gate_name} Gate"
                )
                
                with viz_cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Learning curves
        st.markdown("#### Learning Curves")
        curve_cols = st.columns(2)
        
        for i, gate_name in enumerate(selected_gates):
            if gate_name in st.session_state.trained_results:
                result = st.session_state.trained_results[gate_name]
                
                fig = create_learning_curve_plot(result['history'], gate_name)
                if fig:
                    with curve_cols[i % 2]:
                        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive testing
        st.markdown('<div class="sub-header">🎮 Interactive Testing</div>', unsafe_allow_html=True)
        
        test_col1, test_col2, test_col3 = st.columns([1, 1, 2])
        
        with test_col1:
            test_gate = st.selectbox("Select Gate:", selected_gates)
        
        with test_col2:
            input1 = st.selectbox("Input 1:", [0, 1])
            input2 = st.selectbox("Input 2:", [0, 1])
        
        with test_col3:
            if st.button(" Predict", type="secondary"):
                if test_gate in st.session_state.trained_results:
                    result = st.session_state.trained_results[test_gate]
                    prediction = result['perceptron'].predict_single(np.array([input1, input2]))
                    expected = gates[test_gate][input1 * 2 + input2]
                    
                    if prediction == expected:
                        st.success(f"{test_gate}({input1}, {input2}) = {prediction} (Correct!)")
                    else:
                        st.error(f" {test_gate}({input1}, {input2}) = {prediction} (Expected: {expected})")
    
    else:
        # Welcome message
        st.markdown("""
        <div class="success-box">
        <h3> Welcome to the Perceptron Logic Gate Trainer!</h3>
        <p>This interactive application allows you to:</p>
        <ul>
            <li> Train perceptrons on different logic gates (AND, OR, NAND, XOR)</li>
            <li> Visualize decision boundaries and learning curves</li>
            <li> Experiment with different learning rates and epochs</li>
            <li> Test trained models interactively</li>
        </ul>
        <p><strong>Get started by selecting gates and clicking "Train Perceptrons" in the sidebar!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show logic gate truth tables
        st.markdown("###  Logic Gate Truth Tables")
        
        truth_cols = st.columns(2)
        gates_data = get_all_gates()
        X = get_input_data()
        
        for i, (gate_name, y) in enumerate(gates_data.items()):
            with truth_cols[i % 2]:
                df = pd.DataFrame({
                    'Input 1': X[:, 0].astype(int),
                    'Input 2': X[:, 1].astype(int),
                    f'{gate_name} Output': y
                })
                st.markdown(f"**{gate_name} Gate**")
                st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
