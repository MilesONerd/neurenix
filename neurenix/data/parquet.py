"""
Parquet integration for Neurenix.

This module provides functionality for working with Parquet files,
enabling efficient storage and retrieval of structured data.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Iterator
import warnings

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    warnings.warn(
        "PyArrow not found. Install with 'pip install pyarrow' to use Parquet functionality."
    )

from neurenix.core import get_logger
from neurenix.tensor import Tensor
from neurenix.data.arrow import ArrowTable

logger = get_logger(__name__)

class ParquetDataset:
    """Dataset for reading and writing Parquet files."""
    
    def __init__(self, path: str, columns: Optional[List[str]] = None, filters=None):
        """
        Initialize a ParquetDataset.
        
        Args:
            path: Path to the Parquet file or directory
            columns: Columns to read
            filters: Filters to apply when reading
        """
        if not PARQUET_AVAILABLE:
            raise ImportError(
                "PyArrow is required for ParquetDataset. Install with 'pip install pyarrow'."
            )
        
        self.path = path
        self.columns = columns
        self.filters = filters
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        
        self.is_directory = os.path.isdir(path)
        
        if self.is_directory:
            parquet_files = [f for f in os.listdir(path) if f.endswith('.parquet')]
            if not parquet_files:
                raise ValueError(f"Directory does not contain Parquet files: {path}")
            
            first_file = os.path.join(path, parquet_files[0])
            self.metadata = pq.read_metadata(first_file)
            self.schema = pq.read_schema(first_file)
        else:
            self.metadata = pq.read_metadata(path)
            self.schema = pq.read_schema(path)
    
    @property
    def num_rows(self) -> int:
        """Get the number of rows in the dataset."""
        return self.metadata.num_rows
    
    @property
    def num_columns(self) -> int:
        """Get the number of columns in the dataset."""
        return len(self.schema.names)
    
    @property
    def column_names(self) -> List[str]:
        """Get the column names in the dataset."""
        return self.schema.names
    
    def read(self, columns: Optional[List[str]] = None, filters=None, batch_size: Optional[int] = None) -> ArrowTable:
        """
        Read the dataset into an ArrowTable.
        
        Args:
            columns: Columns to read (overrides columns specified in constructor)
            filters: Filters to apply when reading (overrides filters specified in constructor)
            batch_size: Batch size for reading
            
        Returns:
            ArrowTable
        """
        columns = columns or self.columns
        filters = filters or self.filters
        
        if self.is_directory:
            table = pq.read_table(self.path, columns=columns, filters=filters)
        else:
            table = pq.read_table(self.path, columns=columns, filters=filters)
        
        return ArrowTable(table)
    
    def read_row_group(self, row_group_index: int, columns: Optional[List[str]] = None) -> ArrowTable:
        """
        Read a specific row group from the dataset.
        
        Args:
            row_group_index: Row group index
            columns: Columns to read
            
        Returns:
            ArrowTable
        """
        if self.is_directory:
            raise NotImplementedError("Reading row groups from directories is not supported.")
        
        columns = columns or self.columns
        
        table = pq.read_row_group(self.path, row_group_index, columns=columns)
        
        return ArrowTable(table)
    
    def read_row_groups(self, row_group_indices: List[int], columns: Optional[List[str]] = None) -> ArrowTable:
        """
        Read specific row groups from the dataset.
        
        Args:
            row_group_indices: Row group indices
            columns: Columns to read
            
        Returns:
            ArrowTable
        """
        if self.is_directory:
            raise NotImplementedError("Reading row groups from directories is not supported.")
        
        columns = columns or self.columns
        
        table = pq.read_row_groups(self.path, row_group_indices, columns=columns)
        
        return ArrowTable(table)
    
    def iter_batches(self, batch_size: int, columns: Optional[List[str]] = None, filters=None) -> Iterator[ArrowTable]:
        """
        Iterate over batches of the dataset.
        
        Args:
            batch_size: Batch size
            columns: Columns to read
            filters: Filters to apply when reading
            
        Yields:
            ArrowTable
        """
        columns = columns or self.columns
        filters = filters or self.filters
        
        dataset = pq.ParquetDataset(self.path, filters=filters)
        
        for batch in dataset.iter_batches(batch_size=batch_size, columns=columns):
            yield ArrowTable(pa.Table.from_batches([batch]))
    
    def to_tensors(self, columns: Optional[List[str]] = None) -> Dict[str, Tensor]:
        """
        Convert the dataset to Neurenix Tensors.
        
        Args:
            columns: Columns to convert
            
        Returns:
            Dictionary mapping column names to Neurenix Tensors
        """
        table = self.read(columns=columns)
        
        return table.to_tensors()
    
    @staticmethod
    def write(table: ArrowTable, path: str, compression: str = "snappy", row_group_size: Optional[int] = None,
             version: str = "2.0", write_statistics: bool = True, **kwargs) -> None:
        """
        Write an ArrowTable to a Parquet file.
        
        Args:
            table: ArrowTable
            path: Path to write the Parquet file
            compression: Compression algorithm (snappy, gzip, brotli, zstd, lz4, none)
            row_group_size: Row group size
            version: Parquet format version
            write_statistics: Whether to write statistics
            **kwargs: Additional arguments to pass to pyarrow.parquet.write_table
        """
        if not PARQUET_AVAILABLE:
            raise ImportError(
                "PyArrow is required for write(). Install with 'pip install pyarrow'."
            )
        
        pq.write_table(
            table.table,
            path,
            compression=compression,
            row_group_size=row_group_size,
            version=version,
            write_statistics=write_statistics,
            **kwargs
        )
    
    @staticmethod
    def write_to_dataset(table: ArrowTable, root_path: str, partition_cols: Optional[List[str]] = None,
                        compression: str = "snappy", **kwargs) -> None:
        """
        Write an ArrowTable to a partitioned Parquet dataset.
        
        Args:
            table: ArrowTable
            root_path: Root path for the dataset
            partition_cols: Columns to partition by
            compression: Compression algorithm (snappy, gzip, brotli, zstd, lz4, none)
            **kwargs: Additional arguments to pass to pyarrow.parquet.write_to_dataset
        """
        if not PARQUET_AVAILABLE:
            raise ImportError(
                "PyArrow is required for write_to_dataset(). Install with 'pip install pyarrow'."
            )
        
        pq.write_to_dataset(
            table.table,
            root_path,
            partition_cols=partition_cols,
            compression=compression,
            **kwargs
        )
    
    @staticmethod
    def from_tensors(tensors: Dict[str, Tensor]) -> ArrowTable:
        """
        Create an ArrowTable from a dictionary of Neurenix Tensors.
        
        Args:
            tensors: Dictionary mapping column names to Neurenix Tensors
            
        Returns:
            ArrowTable
        """
        return ArrowTable.from_tensors(tensors)
    
    @staticmethod
    def from_pandas(df):
        """
        Create an ArrowTable from a pandas DataFrame.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            ArrowTable
        """
        return ArrowTable.from_pandas(df)


def read_parquet(path: str, columns: Optional[List[str]] = None, filters=None) -> ArrowTable:
    """
    Read a Parquet file into an ArrowTable.
    
    Args:
        path: Path to the Parquet file or directory
        columns: Columns to read
        filters: Filters to apply when reading
        
    Returns:
        ArrowTable
    """
    if not PARQUET_AVAILABLE:
        raise ImportError(
            "PyArrow is required for read_parquet(). Install with 'pip install pyarrow'."
        )
    
    table = pq.read_table(path, columns=columns, filters=filters)
    
    return ArrowTable(table)


def write_parquet(table: ArrowTable, path: str, compression: str = "snappy", row_group_size: Optional[int] = None,
                 version: str = "2.0", write_statistics: bool = True, **kwargs) -> None:
    """
    Write an ArrowTable to a Parquet file.
    
    Args:
        table: ArrowTable
        path: Path to write the Parquet file
        compression: Compression algorithm (snappy, gzip, brotli, zstd, lz4, none)
        row_group_size: Row group size
        version: Parquet format version
        write_statistics: Whether to write statistics
        **kwargs: Additional arguments to pass to pyarrow.parquet.write_table
    """
    if not PARQUET_AVAILABLE:
        raise ImportError(
            "PyArrow is required for write_parquet(). Install with 'pip install pyarrow'."
        )
    
    pq.write_table(
        table.table,
        path,
        compression=compression,
        row_group_size=row_group_size,
        version=version,
        write_statistics=write_statistics,
        **kwargs
    )


def write_to_dataset(table: ArrowTable, root_path: str, partition_cols: Optional[List[str]] = None,
                    compression: str = "snappy", **kwargs) -> None:
    """
    Write an ArrowTable to a partitioned Parquet dataset.
    
    Args:
        table: ArrowTable
        root_path: Root path for the dataset
        partition_cols: Columns to partition by
        compression: Compression algorithm (snappy, gzip, brotli, zstd, lz4, none)
        **kwargs: Additional arguments to pass to pyarrow.parquet.write_to_dataset
    """
    if not PARQUET_AVAILABLE:
        raise ImportError(
            "PyArrow is required for write_to_dataset(). Install with 'pip install pyarrow'."
        )
    
    pq.write_to_dataset(
        table.table,
        root_path,
        partition_cols=partition_cols,
        compression=compression,
        **kwargs
    )
