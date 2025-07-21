#!/bin/bash

echo "=== GPU/CUDA Fix Script ==="
echo "This script will fix the PyTorch CUDA installation issue."
echo

# Check if NVIDIA driver is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✅ NVIDIA driver found:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

echo
echo "🔧 Checking current PyTorch installation..."
python -c "import torch; print(f'Current PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo
echo "🚀 Installing PyTorch with CUDA support..."

# Uninstall existing PyTorch
echo "Removing existing PyTorch installations..."
pip uninstall torch torchvision torchaudio -y

# Install CUDA-enabled PyTorch
echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo
echo "✅ Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "🎉 SUCCESS! PyTorch can now use GPU."
    echo "You can now run your training scripts with GPU acceleration."
else
    echo "❌ CUDA is still not available. Please check:"
    echo "  1. NVIDIA drivers are properly installed"
    echo "  2. CUDA toolkit is installed"
    echo "  3. Your GPU is supported"
fi
