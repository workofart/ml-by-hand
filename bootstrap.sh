#!/bin/bash
set -euo pipefail

# Trap errors to provide a single, clear message.
trap 'echo "ERROR: Script failed at line $LINENO. Check the output above for more details." >&2' ERR

PYTHON_VERSION="3.12.2"

# 1. Check for required tools
command -v curl >/dev/null 2>&1 || {
    echo "ERROR: 'curl' is required but not found. Please install curl and re-run."
    exit 1
}

# 2. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv from https://astral.sh/uv/install.sh ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "ERROR: 'uv' installation failed. Check your network or try again later."
        exit 1
    fi

    # Update PATH
    export PATH="$HOME/.local/bin:$PATH"
    if ! grep -q "$HOME/.local/bin" <<< "$PATH"; then
        echo "INFO: Add 'export PATH=\"\$HOME/.local/bin:\$PATH\"' to your shell rc to make it permanent."
    fi

    # Source cargo/env if it exists
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    else
        echo "NOTE: $HOME/.cargo/env not found; skipping cargo env source."
    fi
fi

# 3. Create or re-use virtual environment
if [ -d ".venv" ]; then
    if [ -x ".venv/bin/python" ]; then
        INSTALLED_VERSION=$(.venv/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
        if [[ "$INSTALLED_VERSION" == "$PYTHON_VERSION" ]]; then
            echo "Found existing venv with Python $INSTALLED_VERSION."
            echo "Skipping venv creation."
        else
            echo "WARNING: .venv has Python $INSTALLED_VERSION, but we wanted $PYTHON_VERSION."
            echo "You may want to remove or rebuild .venv."
        fi
    else
        echo "WARNING: .venv folder exists but appears corrupted. Remove or fix it before re-running."
    fi
else
    # Install desired Python version
    uv python install "$PYTHON_VERSION" || {
        echo "ERROR: Could not install Python $PYTHON_VERSION via uv."
        exit 1
    }
    uv venv --python "$PYTHON_VERSION"
fi

source .venv/bin/activate
echo "Virtual environment is ready and activated."

# 4. Check GPU and install PyTorch for unit test validation
if [ "$(uname)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
else
    echo "Installing CPU-only PyTorch..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# 5. Install dev dependencies
echo "Installing all dependencies (including dev)..."
if ! uv pip install ".[dev]"; then
    echo "ERROR: Failed to install dev dependencies."
    exit 1
fi

echo "Success! To start using your environment, run:"
echo "   source .venv/bin/activate"
