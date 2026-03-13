#!/bin/bash
# Activate venv with PYTHONPATH set for alpamayo_r1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check which venv exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV="venv"
elif [ -d "$SCRIPT_DIR/ar1_venv" ]; then
    VENV="ar1_venv"
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "❌ No virtual environment found!"
    echo "Run: make venv && make install"
    return 1
fi

# Activate venv
source "$SCRIPT_DIR/$VENV/bin/activate"

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

echo "✓ Activated $VENV with PYTHONPATH set"
echo ""
echo "Now you can run:"
echo "  python demo_inference.py"
echo "  python src/alpamayo_r1/test_inference.py"
