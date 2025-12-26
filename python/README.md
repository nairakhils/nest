# Nest Python Tools

Visualization and analysis scripts for nest hydrodynamics simulations.

## Installation

Requires Python 3.7+ with:
- numpy
- matplotlib

```bash
pip install numpy matplotlib
```

## Usage

### Command-line Interface

```bash
# Plot 1D Sod results
python -m python plot1d sod1d_0001.dat
python -m python plot1d sod1d_*.dat  # Compare multiple times

# Plot 2D Sod as image
python -m python plot2d sod2d_0001.dat --mode image

# Plot 2D Sod x-slice
python -m python plot2d sod2d_xslice_0001.dat --mode slice
python -m python plot2d sod2d_xslice_*.dat --mode slice  # Compare multiple times

# Show plots interactively
python -m python plot1d sod1d_0001.dat --show

# Specify custom output directory
python -m python plot1d sod1d_0001.dat --output figures/custom/
```

### Direct Script Execution

You can also run the scripts directly:

```bash
python python/plot_sod1d.py sod1d_0001.dat
python python/plot_sod2d.py sod2d_0001.dat --mode image
```

### Programmatic API

```python
from python.plot_sod1d import read_sod1d, plot_sod1d
from python.plot_sod2d import read_sod2d, plot_sod2d_image, plot_sod2d_slice

# Read data
data = read_sod1d('sod1d_0001.dat')
print(f"Time: {data['time']}")
print(f"Density range: [{data['rho'].min()}, {data['rho'].max()}]")

# Plot
fig = plot_sod1d(['sod1d_0001.dat', 'sod1d_0002.dat'])
```

## File Formats

### 1D Sod (`sod1d_NNNN.dat`)
```
# 1D Sod shock tube at t = 0.200000
# x rho u p
0.005 0.427683 -0.921602 0.304295
0.015 0.431830 -0.914843 0.308413
...
```

### 2D Sod (`sod2d_NNNN.dat`)
```
# 2D Sod shock tube at t = 0.200000
# x y rho u v p
0.0025 0.0025 0.427683 -0.921602 0.0 0.304295
0.0025 0.0075 0.427683 -0.921602 0.0 0.304295
...
```

### 2D Sod x-slice (`sod2d_xslice_NNNN.dat`)
```
# x-slice at y_mid, t = 0.200000
# x rho u p
0.0025 0.427683 -0.921602 0.304295
0.0075 0.431830 -0.914843 0.308413
...
```

## Output

All figures are saved to `python/figures/` by default (configurable with `--output`).

### 1D Plots
- 3-panel plot showing density, velocity, and pressure vs x
- Multiple times overlaid with color gradient

### 2D Image Plots
- 2x2 grid showing density, u-velocity, v-velocity, and pressure as colormaps

### 2D Slice Plots
- Same as 1D plots but for x-slice at middle y

## Examples

After running the Sod tests:

```bash
# Build and run tests
cd nest
cmake --preset cpu-release
cmake --build build/cpu-release
./build/cpu-release/examples/sod1d/sod1d
./build/cpu-release/examples/sod2d/sod2d

# Plot results
python -m python plot1d sod1d_*.dat
python -m python plot2d sod2d_xslice_*.dat --mode slice
python -m python plot2d sod2d_0001.dat --mode image
```

