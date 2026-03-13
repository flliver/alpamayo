.PHONY: help venv install install-dev clean demo demo-quick test check-hf run-eval

# Python and venv settings
PYTHON := python3.12
VENV := venv
VENV_BIN := $(VENV)/bin
PYTHON_BIN := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

help:
	@echo "Alpamayo R1 - Available commands:"
	@echo ""
	@echo "  make demo-quick   - Run demo using existing ar1_venv (FASTEST)"
	@echo "  make venv         - Create new venv (requires python3.12-venv)"
	@echo "  make install      - Install core dependencies"
	@echo "  make install-dev  - Install with dev dependencies (notebooks)"
	@echo "  make demo         - Run demo with new venv"
	@echo "  make run-eval     - Run full evaluation with real dataset"
	@echo "  make check-hf     - Check HuggingFace access"
	@echo "  make clean        - Remove venv and cache files"
	@echo ""
	@echo "Quickest start (using existing setup):"
	@echo "  make demo-quick"
	@echo ""
	@echo "Fresh install:"
	@echo "  1. sudo apt install python3.12-venv  (if needed)"
	@echo "  2. make venv"
	@echo "  3. make install"
	@echo "  4. make demo"

$(VENV):
	@echo "Creating virtual environment with Python 3.12..."
	@echo "Note: Requires python3.12-venv package on Ubuntu/Debian"
	@echo "      Install with: sudo apt install python3.12-venv"
	@echo ""
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment created at ./$(VENV)"
	@echo ""
	@echo "To activate manually: source $(VENV)/bin/activate"

venv: $(VENV)

install: $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	@echo ""
	@echo "✓ Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run 'make check-hf' to verify HuggingFace access"
	@echo "  - Run 'make demo' to test the model"

install-dev: $(VENV)
	@echo "Installing dependencies with dev extras..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "✓ Installation complete with dev dependencies!"

check-hf: $(VENV)
	@echo "Checking HuggingFace access..."
	@$(PYTHON_BIN) check_access.py

demo-quick:
	@echo "Running demo using existing ar1_venv..."
	@if [ ! -d "ar1_venv" ]; then \
		echo "❌ ar1_venv not found. Run 'make venv && make install' first"; \
		exit 1; \
	fi
	@export PATH="$$HOME/.local/bin:$$PATH" && \
	 . ar1_venv/bin/activate && \
	 python demo_inference.py

demo: $(VENV)
	@echo "Running demo inference with synthetic data..."
	@echo "(Model will download on first run - 22GB, ~5 min)"
	@echo ""
	@$(PYTHON_BIN) demo_inference.py

run-eval: $(VENV)
	@echo "Running evaluation with real dataset..."
	@echo "(Requires HuggingFace access to PhysicalAI-Autonomous-Vehicles)"
	@echo ""
	@$(PYTHON_BIN) src/alpamayo_r1/test_inference.py

test: demo

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf ar1_venv
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned!"

# Shortcuts
.PHONY: i d r
i: install
d: demo
r: run-eval
