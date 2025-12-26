#!/bin/bash
# Demo script for nest python plotting tools

set -e

echo "=== Nest Python Visualization Demo ==="
echo ""

# Check for data files
if [ ! -f sod_0000.dat ]; then
    echo "Error: 1D Sod data not found. Please run:"
    echo "  ./build/cpu-release/examples/sod1d/sod1d"
    exit 1
fi

if [ ! -f sod2d_0000.dat ]; then
    echo "Error: 2D Sod data not found. Please run:"
    echo "  ./build/cpu-release/examples/sod2d/sod2d"
    exit 1
fi

echo "1. Plotting 1D Sod results..."
python3 -m python plot1d sod_0000.dat sod_0004.dat sod_0008.dat

echo ""
echo "2. Plotting 2D Sod x-slices..."
python3 -m python plot2d sod2d_xslice_0000.dat sod2d_xslice_0003.dat sod2d_xslice_0006.dat --mode slice

echo ""
echo "3. Plotting 2D Sod images..."
python3 -m python plot2d sod2d_0000.dat --mode image
python3 -m python plot2d sod2d_0006.dat --mode image

echo ""
echo "=== Demo Complete ==="
echo "Generated figures in python/figures/:"
ls -lh python/figures/*.png

