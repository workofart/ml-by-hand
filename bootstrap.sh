#!/bin/bash

# Install desired Python version and create a virtual environment
PYTHON_VERSION="3.12.2"

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

uv python install $PYTHON_VERSION
uv venv --python $PYTHON_VERSION

echo "Setup complete. Virtual environment is ready."
