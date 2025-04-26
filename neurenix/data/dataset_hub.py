"""
DatasetHub implementation for easy dataset loading and management.
"""

import os
import json
import urllib.request
import urllib.parse
import hashlib
import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import numpy as np
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..binding import get_binding

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatasetFormat(Enum):
    """Supported dataset formats."""
    CSV = auto()
    JSON = auto()
    NUMPY = auto()
    PICKLE = auto()
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    SQL = auto()
    CUSTOM = auto()
    
    @classmethod
    def from_extension(cls, extension: str) -> 'DatasetFormat':
        """Determine format from file extension."""
        extension = extension.lower().lstrip('.')
        if extension in ('csv', 'tsv'):
            return cls.CSV
        elif extension in ('json', 'jsonl'):
            return cls.JSON
        elif extension in ('npy', 'npz'):
            return cls.NUMPY
        elif extension in ('pkl', 'pickle'):
            return cls.PICKLE
        elif extension in ('txt', 'text'):
            return cls.TEXT
        elif extension in ('jpg', 'jpeg', 'png', 'bmp', 'gif'):
            return cls.IMAGE
        elif extension in ('wav', 'mp3', 'ogg', 'flac'):
            return cls.AUDIO
        elif extension in ('mp4', 'avi', 'mov', 'mkv'):
            return cls.VIDEO
        elif extension in ('db', 'sqlite', 'sqlite3'):
            return cls.SQL
        else:
            return cls.CUSTOM


class Dataset:
    """
    Dataset class for handling various data formats and preprocessing.
    """
    
    def __init__(
        self,
        data: Any,
        format: DatasetFormat,
        name: str = None,
        metadata: Dict = None,
        transform: Callable = None
    ):
        """
        Initialize a Dataset object.
        
        Args:
            data: The dataset content
            format: Format of the dataset
            name: Name of the dataset
            metadata: Additional information about the dataset
            transform: Function to transform data samples
        """
        self.data = data
        self.format = format
        self.name = name
        self.metadata = metadata or {}
        self.transform = transform
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if hasattr(self.data, '__len__'):
            return len(self.data)
        return 0
        
    def __getitem__(self, idx: Union[int, slice]) -> Any:
        """Get a sample or batch from the dataset."""
        if hasattr(self.data, '__getitem__'):
            sample = self.data[idx]
            if self.transform is not None:
                if isinstance(idx, slice):
                    return [self.transform(s) for s in sample]
                return self.transform(sample)
            return sample
        raise TypeError(f"Data of type {type(self.data)} does not support indexing")
    
    def to_tensor(self, framework: str = 'auto') -> Any:
        """
        Convert the dataset to a tensor.
        
        Args:
            framework: The framework to use ('torch', 'tensorflow', or 'auto')
            
        Returns:
            Tensor representation of the dataset
        """
        if framework == 'auto':
            if TORCH_AVAILABLE:
                framework = 'torch'
            elif TF_AVAILABLE:
                framework = 'tensorflow'
            else:
                framework = 'numpy'
                
        if framework == 'torch':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available. Install it with 'pip install torch'")
            if isinstance(self.data, np.ndarray):
                return torch.from_numpy(self.data)
            elif isinstance(self.data, pd.DataFrame):
                return torch.from_numpy(self.data.values)
            elif isinstance(self.data, torch.Tensor):
                return self.data
            else:
                raise TypeError(f"Cannot convert data of type {type(self.data)} to torch.Tensor")
                
        elif framework == 'tensorflow':
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not available. Install it with 'pip install tensorflow'")
            if isinstance(self.data, np.ndarray):
                return tf.convert_to_tensor(self.data)
            elif isinstance(self.data, pd.DataFrame):
                return tf.convert_to_tensor(self.data.values)
            elif isinstance(self.data, tf.Tensor):
                return self.data
            else:
                raise TypeError(f"Cannot convert data of type {type(self.data)} to tf.Tensor")
                
        elif framework == 'numpy':
            if isinstance(self.data, np.ndarray):
                return self.data
            elif isinstance(self.data, pd.DataFrame):
                return self.data.values
            elif TORCH_AVAILABLE and isinstance(self.data, torch.Tensor):
                return self.data.detach().cpu().numpy()
            elif TF_AVAILABLE and isinstance(self.data, tf.Tensor):
                return self.data.numpy()
            else:
                raise TypeError(f"Cannot convert data of type {type(self.data)} to numpy.ndarray")
        
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def split(self, ratio: float = 0.8, shuffle: bool = True, seed: int = None) -> Tuple['Dataset', 'Dataset']:
        """
        Split the dataset into training and validation sets.
        
        Args:
            ratio: Proportion of data to use for training (0.0 to 1.0)
            shuffle: Whether to shuffle the data before splitting
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if seed is not None:
            np.random.seed(seed)
            
        n = len(self)
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)
            
        split_idx = int(n * ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        if isinstance(self.data, np.ndarray):
            train_data = self.data[train_indices]
            val_data = self.data[val_indices]
        elif isinstance(self.data, pd.DataFrame):
            train_data = self.data.iloc[train_indices]
            val_data = self.data.iloc[val_indices]
        elif hasattr(self.data, '__getitem__'):
            train_data = [self.data[i] for i in train_indices]
            val_data = [self.data[i] for i in val_indices]
        else:
            raise TypeError(f"Cannot split data of type {type(self.data)}")
            
        train_dataset = Dataset(
            data=train_data,
            format=self.format,
            name=f"{self.name}_train" if self.name else None,
            metadata=self.metadata.copy(),
            transform=self.transform
        )
        
        val_dataset = Dataset(
            data=val_data,
            format=self.format,
            name=f"{self.name}_val" if self.name else None,
            metadata=self.metadata.copy(),
            transform=self.transform
        )
        
        return train_dataset, val_dataset


class DatasetHub:
    """
    Central hub for managing and loading datasets from various sources.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the DatasetHub.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".neurenix", "datasets")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._registered_datasets = {}
        
        try:
            self._phynexus = get_binding()
            self._has_native = hasattr(self._phynexus, 'data') and hasattr(self._phynexus.data, 'load_dataset')
        except ImportError:
            self._phynexus = None
            self._has_native = False
            logger.warning("Phynexus binding not found. Using pure Python implementation.")
    
    def register_dataset(self, name: str, url: str, format: Union[str, DatasetFormat] = None, metadata: Dict = None):
        """
        Register a dataset with the hub.
        
        Args:
            name: Name of the dataset
            url: URL or file path to the dataset
            format: Format of the dataset (auto-detected if None)
            metadata: Additional information about the dataset
        """
        if isinstance(format, str):
            try:
                format = DatasetFormat[format.upper()]
            except KeyError:
                ext = os.path.splitext(url)[1]
                format = DatasetFormat.from_extension(ext)
        
        self._registered_datasets[name] = {
            'url': url,
            'format': format,
            'metadata': metadata or {}
        }
        
    def load_dataset(
        self, 
        source: str, 
        format: Union[str, DatasetFormat] = None,
        transform: Callable = None,
        force_download: bool = False,
        **kwargs
    ) -> Dataset:
        """
        Load a dataset from a URL, file path, or registered name.
        
        Args:
            source: URL, file path, or registered dataset name
            format: Format of the dataset (auto-detected if None)
            transform: Function to transform data samples
            force_download: Whether to force download even if cached
            **kwargs: Additional arguments for specific formats
            
        Returns:
            Dataset object
        """
        if source in self._registered_datasets:
            dataset_info = self._registered_datasets[source]
            url = dataset_info['url']
            format = format or dataset_info['format']
            metadata = dataset_info['metadata']
        else:
            url = source
            metadata = {}
            
        is_remote = url.startswith(('http://', 'https://', 'ftp://'))
        
        if is_remote:
            local_path = self._download_dataset(url, force_download)
        else:
            local_path = url
            
        if format is None:
            ext = os.path.splitext(local_path)[1]
            format = DatasetFormat.from_extension(ext)
            
        if isinstance(format, str):
            try:
                format = DatasetFormat[format.upper()]
            except KeyError:
                raise ValueError(f"Unsupported format: {format}")
                
        if self._has_native and not kwargs.get('disable_native', False):
            try:
                data = self._load_native(local_path, format, **kwargs)
                return Dataset(data, format, name=source, metadata=metadata, transform=transform)
            except Exception as e:
                logger.warning(f"Native loading failed: {e}. Falling back to Python implementation.")
                
        data = self._load_python(local_path, format, **kwargs)
        return Dataset(data, format, name=source, metadata=metadata, transform=transform)
    
    def _download_dataset(self, url: str, force_download: bool = False) -> str:
        """
        Download a dataset from a URL.
        
        Args:
            url: URL of the dataset
            force_download: Whether to force download even if cached
            
        Returns:
            Local path to the downloaded dataset
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        if not filename:
            filename = f"dataset_{url_hash}"
            
        local_path = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(local_path) or force_download:
            logger.info(f"Downloading dataset from {url}")
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset from {url}: {e}")
                
        return local_path
    
    def _load_native(self, path: str, format: DatasetFormat, **kwargs) -> Any:
        """
        Load a dataset using the native Phynexus implementation.
        
        Args:
            path: Path to the dataset
            format: Format of the dataset
            **kwargs: Additional arguments for specific formats
            
        Returns:
            Loaded dataset
        """
        format_str = format.name.lower()
        return self._phynexus.data.load_dataset(path, format_str, json.dumps(kwargs))
    
    def _load_python(self, path: str, format: DatasetFormat, **kwargs) -> Any:
        """
        Load a dataset using the Python implementation.
        
        Args:
            path: Path to the dataset
            format: Format of the dataset
            **kwargs: Additional arguments for specific formats
            
        Returns:
            Loaded dataset
        """
        if format == DatasetFormat.CSV:
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas is required for CSV format. Install it with 'pip install pandas'")
            return pd.read_csv(path, **kwargs)
            
        elif format == DatasetFormat.JSON:
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas is required for JSON format. Install it with 'pip install pandas'")
            return pd.read_json(path, **kwargs)
            
        elif format == DatasetFormat.NUMPY:
            return np.load(path, **kwargs)
            
        elif format == DatasetFormat.PICKLE:
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas is required for pickle format. Install it with 'pip install pandas'")
            return pd.read_pickle(path, **kwargs)
            
        elif format == DatasetFormat.TEXT:
            with open(path, 'r', encoding=kwargs.get('encoding', 'utf-8')) as f:
                return f.read()
                
        elif format == DatasetFormat.IMAGE:
            try:
                from PIL import Image
                return Image.open(path)
            except ImportError:
                raise ImportError("PIL is required for loading images. Install it with 'pip install Pillow'")
                
        elif format == DatasetFormat.AUDIO:
            try:
                import librosa
                return librosa.load(path, **kwargs)
            except ImportError:
                raise ImportError("librosa is required for loading audio. Install it with 'pip install librosa'")
                
        elif format == DatasetFormat.VIDEO:
            try:
                import cv2
                return cv2.VideoCapture(path)
            except ImportError:
                raise ImportError("OpenCV is required for loading videos. Install it with 'pip install opencv-python'")
                
        elif format == DatasetFormat.SQL:
            try:
                import sqlite3
                
                if not PANDAS_AVAILABLE:
                    raise ImportError("pandas is required for SQL format. Install it with 'pip install pandas'")
                
                try:
                    import sqlalchemy
                    has_sqlalchemy = True
                except ImportError:
                    has_sqlalchemy = False
                    logger.warning("SQLAlchemy not found. Using basic sqlite3 functionality.")
                
                if has_sqlalchemy and path.startswith(('sqlite:///', 'mysql://', 'postgresql://', 'oracle://', 'mssql://')):
                    engine = sqlalchemy.create_engine(path)
                    
                    table_name = kwargs.get('table')
                    query = kwargs.get('query')
                    
                    if query:
                        return pd.read_sql_query(query, engine)
                    elif table_name:
                        return pd.read_sql_table(table_name, engine)
                    else:
                        inspector = sqlalchemy.inspect(engine)
                        tables = inspector.get_table_names()
                        if not tables:
                            raise ValueError("No tables found in the database")
                        return pd.read_sql_table(tables[0], engine)
                else:
                    conn = sqlite3.connect(path)
                    
                    table_name = kwargs.get('table')
                    query = kwargs.get('query')
                    
                    if query:
                        return pd.read_sql_query(query, conn)
                    elif table_name:
                        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    else:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        if not tables:
                            raise ValueError("No tables found in the database")
                        return pd.read_sql_query(f"SELECT * FROM {tables[0][0]}", conn)
            except ImportError as e:
                raise ImportError(f"Required dependency missing for SQL support: {str(e)}. Install with 'pip install pandas sqlalchemy'")
                
        else:
            raise ValueError(f"Unsupported format: {format}")
