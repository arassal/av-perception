#!/bin/bash
# YOLOPv2 + YOLOv8 Setup Script

echo "=========================================="
echo "YOLOPv2 + YOLOv8 Perception System Setup"
echo "=========================================="
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

echo "✓ Python found: $(python --version)"
echo ""

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Create directories
echo "📁 Creating required directories..."
mkdir -p data/weights
mkdir -p images
mkdir -p runs/detect

echo "✓ Directories created"
echo ""

# Check for models
echo "🤖 Checking models..."

if [ ! -f "data/weights/yolopv2.pt" ]; then
    echo "⚠️  YOLOPv2 model not found at data/weights/yolopv2.pt"
    echo "   Please download from: https://github.com/CAIC-AD/YOLOPv2/releases"
else
    echo "✓ YOLOPv2 model found"
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To start the GUI:"
echo "  python yolopv2_combined_gui.py"
echo ""
echo "To test camera setup:"
echo "  python check_camera.py"
echo ""
echo "For help:"
echo "  See GUI_README.md"
echo ""
