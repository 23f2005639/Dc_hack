#!/usr/bin/env python3
"""
Launcher script for the Perceptron Logic Gate Trainer Streamlit App
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    print("Starting Perceptron Logic Gate Trainer...")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "streamlit_app.py")
    
    try:
        # Launch Streamlit
        subprocess.run([
            "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication closed by user")
    except FileNotFoundError:
        print(" Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
