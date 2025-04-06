"""
DatasetHub module for easy dataset loading and management.

This module provides utilities for loading datasets from URLs or file paths,
supporting various formats and preprocessing options.
"""

from .dataset_hub import DatasetHub, Dataset, DatasetFormat

def load_dataset(*args, **kwargs):
    """
    Convenience function to load a dataset using the default DatasetHub instance.
    
    See DatasetHub.load_dataset for full documentation.
    """
    hub = DatasetHub()
    return hub.load_dataset(*args, **kwargs)

def register_dataset(*args, **kwargs):
    """
    Convenience function to register a dataset using the default DatasetHub instance.
    
    See DatasetHub.register_dataset for full documentation.
    """
    hub = DatasetHub()
    return hub.register_dataset(*args, **kwargs)

__all__ = [
    'DatasetHub',
    'Dataset',
    'DatasetFormat',
    'load_dataset',
    'register_dataset'
]
