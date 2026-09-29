#!/bin/bash

# Clear the terminal
clear

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the shared venv (lives at the repo root so other models can reuse it)
source "$SCRIPT_DIR/../.venv-python-model/bin/activate" || { echo "Failed to activate venv"; exit 1; }

# Compile with ninja
cd "$SCRIPT_DIR/../jupedsim/build" || { echo "Failed to change directory"; exit 1; }

echo "Compiling with ninja..."
if ! ninja; then
    echo "Compilation failed!"
    exit 1
fi

# Source the environment file to set PYTHONPATH to local build
if [ -f environment ]; then
    source environment
else
    echo "Environment file not found!"
    exit 1
fi

# Check if a Python file name was provided
if [ -z "$1" ]; then
    echo "No simulation provided."
    exit 1
fi

# Execute the Python file
cd "$SCRIPT_DIR" || { echo "Failed to change directory"; exit 1; }
python "$1"