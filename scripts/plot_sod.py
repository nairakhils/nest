#!/usr/bin/env python3
"""
Plot Sod shock tube results from nest output files.

Usage:
    python plot_sod.py sod_*.dat
    python plot_sod.py sod_0000.dat sod_0010.dat  # Compare specific times
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_sod_file(filepath: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read Sod output file and return (time, x, rho, u, p, e)."""
    time = 0.0
    x, rho, u, p, e = [], [], [], [], []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '# Sod shock tube at t =' in line:
                time = float(line.split('=')[1].strip())
            elif line.startswith('#'):
                continue
            else:
                parts = line.split()
                if len(parts) >= 5:
                    x.append(float(parts[0]))
                    rho.append(float(parts[1]))
                    u.append(float(parts[2]))
                    p.append(float(parts[3]))
                    e.append(float(parts[4]))
    
    return time, np.array(x), np.array(rho), np.array(u), np.array(p), np.array(e)


def plot_sod_comparison(files: list[Path]):
    """Plot multiple Sod shock tube results for comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    
    for filepath, color in zip(files, colors):
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist, skipping")
            continue
        
        time, x, rho, u, p, e = read_sod_file(filepath)
        label = f't = {time:.3f}'
        
        axes[0, 0].plot(x, rho, '-', color=color, label=label, linewidth=1.5)
        axes[0, 1].plot(x, u, '-', color=color, label=label, linewidth=1.5)
        axes[1, 0].plot(x, p, '-', color=color, label=label, linewidth=1.5)
        axes[1, 1].plot(x, e, '-', color=color, label=label, linewidth=1.5)
    
    # Density
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('ρ')
    axes[0, 0].set_title('Density')
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 1)
    
    # Velocity
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('u')
    axes[0, 1].set_title('Velocity')
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    
    # Pressure
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('p')
    axes[1, 0].set_title('Pressure')
    axes[1, 0].legend(loc='best', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    
    # Internal energy
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('e')
    axes[1, 1].set_title('Specific Internal Energy')
    axes[1, 1].legend(loc='best', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    
    plt.suptitle('Sod Shock Tube Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sod_plot.png', dpi=150)
    print('Saved sod_plot.png')
    plt.show()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    files = [Path(f) for f in sys.argv[1:]]
    files.sort(key=lambda p: p.name)
    
    plot_sod_comparison(files)


if __name__ == '__main__':
    main()

