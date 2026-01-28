#!/bin/bash
# Automated setup script for Ubuntu after account creation
# Run this inside Ubuntu after you've created your account

echo "========================================"
echo "PhishSim - Automated Build Setup"
echo "========================================"
echo ""

# Update package list
echo "Step 1/4: Updating package list..."
sudo apt update

echo ""
echo "Step 2/4: Installing build dependencies..."
echo "  - g++ (C++ compiler)"
echo "  - liblzma-dev (LZMA compression library)"
echo "  - libxxhash-dev (xxHash library)"
echo "  - build-essential (build tools)"
echo ""
sudo apt install -y g++ liblzma-dev libxxhash-dev build-essential

echo ""
echo "Step 3/4: Navigating to project directory..."
cd /mnt/c/Users/Nagireddy123/Desktop/Project-f/phishsim

echo ""
echo "Step 4/4: Building C++ NCD worker..."
bash build.sh

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Verification:"
ls -lh ncd_worker 2>/dev/null && echo "✓ C++ worker built successfully!" || echo "✗ Build failed - check errors above"

echo ""
echo "To run fast clustering:"
echo "  python fast_cluster.py"
echo ""
