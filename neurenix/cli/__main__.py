"""
Entry point for the Neurenix CLI.

This module allows the CLI to be run as a Python module:
python -m neurenix.cli <command> [<args>]
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
