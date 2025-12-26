#!/usr/bin/env python3
"""
Main entry point for nest python package.

Usage:
    python -m python <command> [args]
    
Commands:
    plot1d      Plot 1D Sod shock tube results
    plot2d      Plot 2D Sod shock tube results
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Nest visualization and analysis tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  plot1d      Plot 1D Sod shock tube results
  plot2d      Plot 2D Sod shock tube results

Examples:
  python -m python plot1d sod1d_0001.dat
  python -m python plot2d sod2d_0001.dat --mode image
  python -m python plot2d sod2d_xslice_*.dat --mode slice

Run 'python -m python <command> --help' for more information on a command.
        """
    )
    parser.add_argument('command', choices=['plot1d', 'plot2d'],
                        help='Command to run')
    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help='Arguments for the command')
    
    args = parser.parse_args()
    
    if args.command == 'plot1d':
        from .plot_sod1d import main as plot1d_main
        sys.argv = [f'python -m python.{args.command}'] + args.args
        return plot1d_main()
    elif args.command == 'plot2d':
        from .plot_sod2d import main as plot2d_main
        sys.argv = [f'python -m python.{args.command}'] + args.args
        return plot2d_main()
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

