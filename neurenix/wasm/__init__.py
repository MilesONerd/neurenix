"""
WebAssembly module for the Neurenix framework.

This module provides functionality for running Neurenix models in the browser
using WebAssembly.
"""

from .browser import run_in_browser, export_to_wasm

__all__ = [
    'run_in_browser',
    'export_to_wasm',
]
