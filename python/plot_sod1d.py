#!/usr/bin/env python3
"""
Plot 1D Sod shock tube results.

Usage:
    python -m python.plot_sod1d <data_file> [--output <figure_path>]
    python -m python.plot_sod1d sod1d_0001.dat
    python -m python.plot_sod1d sod1d_*.dat  # Plot multiple files
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def read_sod1d(filepath):
    """Read 1D Sod data file."""
    with open(filepath, 'r') as f:
        # Read header to get time
        header = f.readline().strip()
        if 't =' in header:
            time = float(header.split('t =')[1].strip())
        else:
            time = 0.0
        
        # Skip column names
        f.readline()
        
        # Read data
        data = np.loadtxt(f)
    
    return {
        'x': data[:, 0],
        'rho': data[:, 1],
        'u': data[:, 2],
        'p': data[:, 3],
        'time': time
    }


def plot_sod1d(files, output_dir=None):
    """Plot 1D Sod shock tube results."""
    if not isinstance(files, list):
        files = [files]
    
    # Sort files by name (assumes numeric ordering)
    files = sorted(files, key=lambda f: Path(f).name)
    
    # Set up figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle('1D Sod Shock Tube', fontsize=14, fontweight='bold')
    
    # Color map for multiple times
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    
    for i, filepath in enumerate(files):
        data = read_sod1d(filepath)
        color = colors[i]
        label = f't = {data["time"]:.3f}'
        
        # Density
        axes[0].plot(data['x'], data['rho'], color=color, label=label, linewidth=1.5)
        axes[0].set_ylabel(r'Density $\rho$', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Velocity
        axes[1].plot(data['x'], data['u'], color=color, label=label, linewidth=1.5)
        axes[1].set_ylabel(r'Velocity $u$', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Pressure
        axes[2].plot(data['x'], data['p'], color=color, label=label, linewidth=1.5)
        axes[2].set_ylabel(r'Pressure $p$', fontsize=11)
        axes[2].set_xlabel('x', fontsize=11)
        axes[2].grid(True, alpha=0.3)
    
    # Add legends
    for ax in axes:
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir is None:
        output_dir = Path(__file__).parent / 'figures'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(files) == 1:
        output_file = output_dir / f'sod1d_{Path(files[0]).stem}.png'
    else:
        output_file = output_dir / 'sod1d_comparison.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot 1D Sod shock tube results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m python.plot_sod1d sod1d_0001.dat
  python -m python.plot_sod1d sod1d_*.dat
  python -m python.plot_sod1d sod1d_0001.dat --output figures/custom.png
        """
    )
    parser.add_argument('files', nargs='+', help='Data file(s) to plot')
    parser.add_argument('--output', '-o', help='Output directory (default: python/figures/)')
    parser.add_argument('--show', action='store_true', help='Display plot interactively')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    files = []
    for pattern in args.files:
        matches = list(Path('.').glob(pattern))
        if matches:
            files.extend([str(f) for f in matches])
        else:
            # Try as literal filename
            if Path(pattern).exists():
                files.append(pattern)
            else:
                print(f"Warning: File not found: {pattern}", file=sys.stderr)
    
    if not files:
        print("Error: No valid files found", file=sys.stderr)
        return 1
    
    print(f"Plotting {len(files)} file(s)...")
    plot_sod1d(files, output_dir=args.output)
    
    if args.show:
        plt.show()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

