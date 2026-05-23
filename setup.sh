#!/bin/bash

set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo "Installing Detectron2..."
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+18f6958pt2.8.0cu129

echo "Installing benchmark dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete."