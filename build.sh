#!/bin/bash
echo "Building ultra-fast NCD worker..."
g++ -O3 -march=native -flto -fopenmp -pthread \
    ncd_worker.cpp -llzma -lxxhash -o ncd_worker

if [ $? -eq 0 ]; then
    echo "Build successful! → ./ncd_worker"
    chmod +x ncd_worker
else
    echo "Build failed!"
    echo "Install missing packages:"
    echo "   Ubuntu/Debian: sudo apt install liblzma-dev libxxhash-dev g++"
    echo "   macOS:         brew install xz xxhash"
    echo "   Fedora:        sudo dnf install xz-devel xxhash-devel gcc-c++"
fi