#!/bin/bash
# Development installation script for JEPA

set -e  # Exit on any error

echo "🚀 JEPA Development Installation Script"
echo "======================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed or not in PATH"
    exit 1
fi

echo "✅ pip found: $(pip --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install in development mode
echo "🔧 Installing JEPA in development mode..."
pip install -e .

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install -e ".[dev]"

# Install documentation dependencies
echo "🔧 Installing documentation dependencies..."
pip install -e ".[docs]"

echo ""
echo "✅ Installation complete!"
echo ""
echo "🔍 Verifying installation..."
python verify_install.py

echo ""
echo "🎉 JEPA development environment is ready!"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To train a model:"
echo "  jepa-train --config config/default_config.yaml"
