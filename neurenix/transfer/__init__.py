"""
Transfer learning module for the Neurenix framework.

This module provides tools and utilities for transfer learning, allowing users
to leverage pre-trained models and adapt them to new tasks with minimal data.
"""

from neurenix.transfer.model import TransferModel
from neurenix.transfer.fine_tuning import fine_tune, freeze_layers, unfreeze_layers

__all__ = [
    'TransferModel',
    'fine_tune',
    'freeze_layers',
    'unfreeze_layers',
]
