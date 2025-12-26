#!/usr/bin/env python3
"""
Plot 1D advection results from nest output files.

Usage:
    python plot_advect1d.py advect1d_*.dat
    python plot_advect1d.py advect1d_0000.dat advect1d_0010.dat  # Compare specific files
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_data_file(filepath: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Read a nest output file and return (time, x, u)."""
    time = 0.0
    x_data = []
    u_data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('# time ='):
                time = float(line.split('=')[1].strip())
            elif line.startswith('#'):
                continue
            else:
                parts = line.split()
                if len(parts) >= 2:
                    x_data.append(float(parts[0]))
                    u_data.append(float(parts[1]))
    
    return time, np.array(x_data), np.array(u_data)


def plot_single(filepath: Path, ax: plt.Axes) -> None:
    """Plot a single data file."""
    time, x, u = read_data_file(filepath)
    ax.plot(x, u, 'o-', markersize=2, label=f't = {time:.4f}')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    files = [Path(f) for f in sys.argv[1:]]
    
    # Sort by filename (assumes numbered output)
    files.sort(key=lambda p: p.name)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for filepath in files:
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist, skipping")
            continue
        plot_single(filepath, ax)
    
    # Plot exact solution at t=1 (one full period)
    x_exact = np.linspace(0, 1, 500)
    u_exact = np.sin(2 * np.pi * x_exact)
    ax.plot(x_exact, u_exact, 'k--', alpha=0.5, label='Exact (t=1)')
    
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('1D Linear Advection')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig('advect1d_plot.png', dpi=150)
    print('Saved advect1d_plot.png')
    plt.show()


if __name__ == '__main__':
    main()



