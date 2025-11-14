#!/bin/bash

# download.sh
# Install Python and system dependencies for SEGMA.

set -e

echo "=== SEGMA Dependency Installer ==="

# Step 1: Install Python packages
echo "[1/2] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 2: Install Foldseek via conda
echo "[2/2] Installing Foldseek using conda..."
conda install bioconda::foldseek -y

echo "=== All dependencies installed successfully! ==="