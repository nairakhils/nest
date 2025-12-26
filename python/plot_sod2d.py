#!/usr/bin/env python3
"""
Plot 2D Sod shock tube results.

Usage:
    python -m python.plot_sod2d <data_file> [--mode slice|image|both]
    python -m python.plot_sod2d sod2d_0001.dat --mode image
    python -m python.plot_sod2d sod2d_xslice_*.dat --mode slice
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm


def read_sod2d(filepath):
    """Read 2D Sod data file."""
    with open(filepath, 'r') as f:
        # Read header to get time
        header = f.readline().strip()
        if 't =' in header:
            time = float(header.split('t =')[1].strip())
        else:
            time = 0.0
        
        # Read column names
        columns = f.readline().strip()
        
        # Read data
        data = np.loadtxt(f)
    
    if data.shape[1] == 6:  # Full 2D: x y rho u v p
        return {
            'x': data[:, 0],
            'y': data[:, 1],
            'rho': data[:, 2],
            'u': data[:, 3],
            'v': data[:, 4],
            'p': data[:, 5],
            'time': time,
            'ndim': 2
        }
    elif data.shape[1] == 4:  # 1D slice: x rho u p
        return {
            'x': data[:, 0],
            'rho': data[:, 1],
            'u': data[:, 2],
            'p': data[:, 3],
            'time': time,
            'ndim': 1
        }
    else:
        raise ValueError(f"Unexpected data format: {data.shape}")


def plot_sod2d_slice(files, output_dir=None):
    """Plot 2D Sod x-slice (like 1D)."""
    if not isinstance(files, list):
        files = [files]
    
    files = sorted(files, key=lambda f: Path(f).name)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle('2D Sod Shock Tube (x-slice at y=middle)', fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    
    for i, filepath in enumerate(files):
        data = read_sod2d(filepath)
        if data['ndim'] != 1:
            print(f"Warning: Skipping {filepath} (not a 1D slice)", file=sys.stderr)
            continue
        
        color = colors[i]
        label = f't = {data["time"]:.3f}'
        
        axes[0].plot(data['x'], data['rho'], color=color, label=label, linewidth=1.5)
        axes[0].set_ylabel(r'Density $\rho$', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(data['x'], data['u'], color=color, label=label, linewidth=1.5)
        axes[1].set_ylabel(r'Velocity $u$', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(data['x'], data['p'], color=color, label=label, linewidth=1.5)
        axes[2].set_ylabel(r'Pressure $p$', fontsize=11)
        axes[2].set_xlabel('x', fontsize=11)
        axes[2].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / 'figures'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(files) == 1:
        output_file = output_dir / f'sod2d_slice_{Path(files[0]).stem}.png'
    else:
        output_file = output_dir / 'sod2d_slice_comparison.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def plot_sod2d_image(filepath, output_dir=None):
    """Plot 2D Sod as images."""
    data = read_sod2d(filepath)
    
    if data['ndim'] != 2:
        raise ValueError("Not a 2D data file")
    
    # Reconstruct 2D grid
    x_unique = np.unique(data['x'])
    y_unique = np.unique(data['y'])
    nx, ny = len(x_unique), len(y_unique)
    
    # Reshape data to 2D arrays
    rho_2d = data['rho'].reshape((ny, nx), order='F')
    u_2d = data['u'].reshape((ny, nx), order='F')
    v_2d = data['v'].reshape((ny, nx), order='F')
    p_2d = data['p'].reshape((ny, nx), order='F')
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'2D Sod Shock Tube at t = {data["time"]:.3f}', 
                 fontsize=14, fontweight='bold')
    
    extent = [x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]]
    
    # Density
    im0 = axes[0, 0].imshow(rho_2d, extent=extent, origin='lower', 
                            aspect='auto', cmap='viridis')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title(r'Density $\rho$')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Velocity u
    im1 = axes[0, 1].imshow(u_2d, extent=extent, origin='lower', 
                            aspect='auto', cmap='RdBu_r')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_title(r'Velocity $u$')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Velocity v
    im2 = axes[1, 0].imshow(v_2d, extent=extent, origin='lower', 
                            aspect='auto', cmap='RdBu_r')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title(r'Velocity $v$')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Pressure
    im3 = axes[1, 1].imshow(p_2d, extent=extent, origin='lower', 
                            aspect='auto', cmap='plasma')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title(r'Pressure $p$')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / 'figures'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'sod2d_image_{Path(filepath).stem}.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot 2D Sod shock tube results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot 2D image
  python -m python.plot_sod2d sod2d_0001.dat --mode image
  
  # Plot x-slice comparison
  python -m python.plot_sod2d sod2d_xslice_*.dat --mode slice
  
  # Plot both image and slice
  python -m python.plot_sod2d sod2d_0001.dat --mode both
        """
    )
    parser.add_argument('files', nargs='+', help='Data file(s) to plot')
    parser.add_argument('--mode', choices=['slice', 'image', 'both'], default='both',
                        help='Plot mode (default: both)')
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
            if Path(pattern).exists():
                files.append(pattern)
            else:
                print(f"Warning: File not found: {pattern}", file=sys.stderr)
    
    if not files:
        print("Error: No valid files found", file=sys.stderr)
        return 1
    
    print(f"Processing {len(files)} file(s)...")
    
    # Determine mode based on files if not explicitly set
    if args.mode == 'both':
        # Check first file to determine type
        data = read_sod2d(files[0])
        if data['ndim'] == 1:
            args.mode = 'slice'
        else:
            args.mode = 'image'
    
    if args.mode == 'slice':
        plot_sod2d_slice(files, output_dir=args.output)
    elif args.mode == 'image':
        for filepath in files:
            plot_sod2d_image(filepath, output_dir=args.output)
    
    if args.show:
        plt.show()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

