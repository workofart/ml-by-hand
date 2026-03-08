#!/bin/bash
set -euo pipefail

# Trap errors to provide a single, clear message.
trap 'echo "ERROR: Script failed at line $LINENO. Check the output above for more details." >&2' ERR

PYTHON_VERSION="3.12.2"

detect_cuda_major_version() {
    local version=""

    if command -v nvcc >/dev/null 2>&1; then
        version=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)
    elif command -v nvidia-smi >/dev/null 2>&1; then
        version=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\..*/\1/p' | head -n 1)
    fi

    if [[ -n "$version" ]]; then
        printf '%s\n' "$version"
        return 0
    fi

    return 1
}

cupy_extra_for_cuda_major() {
    case "$1" in
        11) printf 'cuda11\n' ;;
        12) printf 'cuda12\n' ;;
        13) printf 'cuda13\n' ;;
        *) return 1 ;;
    esac
}

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

# 4. Sync locked project dependencies
SYNC_ARGS=(sync --frozen --extra dev)
CUPY_EXTRA=""
CUDA_MAJOR=""

if [[ "$(uname)" == "Linux" ]] && CUDA_MAJOR=$(detect_cuda_major_version); then
    if CUPY_EXTRA=$(cupy_extra_for_cuda_major "$CUDA_MAJOR"); then
        echo "Detected CUDA $CUDA_MAJOR; syncing optional CuPy extra '$CUPY_EXTRA'."
        SYNC_ARGS+=(--extra "$CUPY_EXTRA")
    else
        echo "Detected CUDA $CUDA_MAJOR, but no pinned CuPy extra is configured for it."
        echo "Proceeding without CuPy. Install the matching CuPy wheel manually if needed."
    fi
fi

echo "Syncing project dependencies from uv.lock..."
if ! uv "${SYNC_ARGS[@]}"; then
    echo "ERROR: Failed to sync project dependencies from uv.lock."
    exit 1
fi

# 5. Install PyTorch separately for unit test validation
echo "Installing CPU-only PyTorch for test validation..."
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Success! To start using your environment, run:"
echo "   source .venv/bin/activate"
echo "Backend selection is automatic; set AUTOGRAD_BACKEND=numpy, AUTOGRAD_BACKEND=mlx, or AUTOGRAD_BACKEND=cupy to override."
