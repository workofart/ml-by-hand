#!/bin/bash
set -e

# Install desired Python version and create a virtual environment
PYTHON_VERSION="3.12.2"

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Check if venv exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping creating venv."
else
    uv python install $PYTHON_VERSION
    uv venv --python $PYTHON_VERSION
fi

echo "Setup complete. Virtual environment is ready."

source .venv/bin/activate

echo "Installing dependencies..."

# Install PyTorch with CUDA support if on Linux with NVIDIA drivers
# PyTorch is only used for unit test validation
if [ "$(uname)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
else
    echo "No NVIDIA GPU detected. Installing PyTorch without CUDA support..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# echo "Installing remaining requirements"
uv pip install -r requirements.txt

