"""
Apache Arrow integration for Neurenix.

This module provides functionality for working with Apache Arrow data structures,
enabling efficient in-memory data processing and interoperability with other
data processing frameworks.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Iterator
import warnings

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False
    warnings.warn(
        "PyArrow not found. Install with 'pip install pyarrow' to use Arrow functionality."
    )

from neurenix.core import get_logger
from neurenix.tensor import Tensor

logger = get_logger(__name__)

class ArrowTable:
    """Wrapper for PyArrow Table with integration to Neurenix tensors."""
    
    def __init__(self, data: Union[pa.Table, Dict[str, Any], List[Dict[str, Any]], None] = None):
        """
        Initialize an Arrow table.
        
        Args:
            data: Data to initialize the table with. Can be a PyArrow Table,
                 a dictionary mapping column names to values, or a list of dictionaries.
        """
        if not ARROW_AVAILABLE:
            raise ImportError(
                "PyArrow is required for ArrowTable. Install with 'pip install pyarrow'."
            )
        
        if data is None:
            self._table = pa.table({})
        elif isinstance(data, pa.Table):
            self._table = data
        elif isinstance(data, dict):
            self._table = pa.table(data)
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            self._table = pa.Table.from_pylist(data)
        else:
            raise ValueError(
                "Data must be a PyArrow Table, a dictionary, or a list of dictionaries."
            )
    
    @property
    def table(self) -> pa.Table:
        """Get the underlying PyArrow Table."""
        return self._table
    
    @property
    def schema(self) -> pa.Schema:
        """Get the schema of the table."""
        return self._table.schema
    
    @property
    def column_names(self) -> List[str]:
        """Get the column names of the table."""
        return self._table.column_names
    
    @property
    def num_rows(self) -> int:
        """Get the number of rows in the table."""
        return self._table.num_rows
    
    @property
    def num_columns(self) -> int:
        """Get the number of columns in the table."""
        return self._table.num_columns
    
    def column(self, name_or_index: Union[str, int]) -> pa.ChunkedArray:
        """
        Get a column from the table.
        
        Args:
            name_or_index: Column name or index
            
        Returns:
            Column as a PyArrow ChunkedArray
        """
        return self._table.column(name_or_index)
    
    def select(self, columns: List[Union[str, int]]) -> 'ArrowTable':
        """
        Select columns from the table.
        
        Args:
            columns: List of column names or indices
            
        Returns:
            New ArrowTable with selected columns
        """
        return ArrowTable(self._table.select(columns))
    
    def filter(self, mask: Union[pa.Array, pa.ChunkedArray, List[bool], np.ndarray]) -> 'ArrowTable':
        """
        Filter rows based on a boolean mask.
        
        Args:
            mask: Boolean mask
            
        Returns:
            New ArrowTable with filtered rows
        """
        return ArrowTable(self._table.filter(mask))
    
    def to_tensor(self, column: Union[str, int]) -> Tensor:
        """
        Convert a column to a Neurenix Tensor.
        
        Args:
            column: Column name or index
            
        Returns:
            Column data as a Neurenix Tensor
        """
        col_data = self.column(column).to_numpy()
        return Tensor(col_data)
    
    def to_tensors(self) -> Dict[str, Tensor]:
        """
        Convert all columns to Neurenix Tensors.
        
        Returns:
            Dictionary mapping column names to Neurenix Tensors
        """
        return {name: self.to_tensor(name) for name in self.column_names}
    
    @staticmethod
    def from_tensor(tensor: Tensor, name: str = "data") -> 'ArrowTable':
        """
        Create an ArrowTable from a Neurenix Tensor.
        
        Args:
            tensor: Neurenix Tensor
            name: Column name
            
        Returns:
            New ArrowTable
        """
        return ArrowTable({name: tensor.numpy()})
    
    @staticmethod
    def from_tensors(tensors: Dict[str, Tensor]) -> 'ArrowTable':
        """
        Create an ArrowTable from a dictionary of Neurenix Tensors.
        
        Args:
            tensors: Dictionary mapping column names to Neurenix Tensors
            
        Returns:
            New ArrowTable
        """
        return ArrowTable({name: tensor.numpy() for name, tensor in tensors.items()})
    
    def to_pandas(self):
        """
        Convert to a pandas DataFrame.
        
        Returns:
            pandas DataFrame
        """
        try:
            return self._table.to_pandas()
        except ImportError:
            raise ImportError(
                "Pandas is required for to_pandas(). Install with 'pip install pandas'."
            )
    
    @staticmethod
    def from_pandas(df):
        """
        Create an ArrowTable from a pandas DataFrame.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            New ArrowTable
        """
        try:
            import pandas as pd
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame.")
            return ArrowTable(pa.Table.from_pandas(df))
        except ImportError:
            raise ImportError(
                "Pandas is required for from_pandas(). Install with 'pip install pandas'."
            )
    
    def to_pylist(self) -> List[Dict[str, Any]]:
        """
        Convert to a list of dictionaries.
        
        Returns:
            List of dictionaries
        """
        return self._table.to_pylist()
    
    def take(self, indices: Union[List[int], np.ndarray, pa.Array]) -> 'ArrowTable':
        """
        Take rows at the given indices.
        
        Args:
            indices: Row indices
            
        Returns:
            New ArrowTable with selected rows
        """
        return ArrowTable(self._table.take(indices))
    
    def slice(self, offset: int, length: Optional[int] = None) -> 'ArrowTable':
        """
        Slice the table.
        
        Args:
            offset: Starting row index
            length: Number of rows to include
            
        Returns:
            New ArrowTable with sliced rows
        """
        return ArrowTable(self._table.slice(offset, length))
    
    def add_column(self, index: int, field_: Union[str, pa.Field], column: Union[pa.Array, pa.ChunkedArray, List, np.ndarray]) -> 'ArrowTable':
        """
        Add a column to the table.
        
        Args:
            index: Position to add the column
            field_: Column name or PyArrow Field
            column: Column data
            
        Returns:
            New ArrowTable with added column
        """
        return ArrowTable(self._table.add_column(index, field_, column))
    
    def append_column(self, field_: Union[str, pa.Field], column: Union[pa.Array, pa.ChunkedArray, List, np.ndarray]) -> 'ArrowTable':
        """
        Append a column to the table.
        
        Args:
            field_: Column name or PyArrow Field
            column: Column data
            
        Returns:
            New ArrowTable with appended column
        """
        return ArrowTable(self._table.append_column(field_, column))
    
    def remove_column(self, index: int) -> 'ArrowTable':
        """
        Remove a column from the table.
        
        Args:
            index: Column index
            
        Returns:
            New ArrowTable with removed column
        """
        return ArrowTable(self._table.remove_column(index))
    
    def cast(self, target_schema: pa.Schema) -> 'ArrowTable':
        """
        Cast the table to a target schema.
        
        Args:
            target_schema: Target schema
            
        Returns:
            New ArrowTable with cast columns
        """
        return ArrowTable(self._table.cast(target_schema))
    
    def group_by(self, keys: List[str]) -> 'ArrowGroupBy':
        """
        Group the table by the given keys.
        
        Args:
            keys: Column names to group by
            
        Returns:
            ArrowGroupBy object
        """
        return ArrowGroupBy(self, keys)
    
    def join(self, right: 'ArrowTable', keys: List[str], join_type: str = "inner") -> 'ArrowTable':
        """
        Join with another table.
        
        Args:
            right: Right table
            keys: Join keys
            join_type: Join type (inner, outer, left, right, full)
            
        Returns:
            Joined ArrowTable
        """
        try:
            import pyarrow.compute as pc
            result = pc.join(self._table, right._table, keys, join_type)
            return ArrowTable(result)
        except (ImportError, AttributeError):
            raise NotImplementedError(
                "Join operation requires PyArrow with compute module."
            )
    
    def to_batches(self, max_chunksize: Optional[int] = None) -> List[pa.RecordBatch]:
        """
        Convert to a list of record batches.
        
        Args:
            max_chunksize: Maximum chunk size
            
        Returns:
            List of record batches
        """
        return self._table.to_batches(max_chunksize)
    
    @staticmethod
    def from_batches(batches: List[pa.RecordBatch], schema: Optional[pa.Schema] = None) -> 'ArrowTable':
        """
        Create an ArrowTable from a list of record batches.
        
        Args:
            batches: List of record batches
            schema: Schema
            
        Returns:
            New ArrowTable
        """
        return ArrowTable(pa.Table.from_batches(batches, schema))
    
    def to_arrow(self) -> pa.Table:
        """
        Get the underlying PyArrow Table.
        
        Returns:
            PyArrow Table
        """
        return self._table
    
    def __len__(self) -> int:
        """Get the number of rows in the table."""
        return self.num_rows
    
    def __getitem__(self, key: Union[str, int, slice, List[Union[str, int]]]) -> Union['ArrowTable', pa.ChunkedArray]:
        """
        Get a column or subset of the table.
        
        Args:
            key: Column name, index, slice, or list of column names/indices
            
        Returns:
            Column or subset of the table
        """
        if isinstance(key, (str, int)):
            return self.column(key)
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.num_rows)
            if step != 1:
                indices = list(range(start, stop, step))
                return self.take(indices)
            return self.slice(start, stop - start)
        elif isinstance(key, list):
            return self.select(key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rows as dictionaries."""
        for batch in self.to_batches():
            for i in range(len(batch)):
                yield {name: batch.column(name)[i].as_py() for name in batch.schema.names}
    
    def __repr__(self) -> str:
        """String representation of the table."""
        return f"ArrowTable({self._table})"


class ArrowGroupBy:
    """Group by operation for ArrowTable."""
    
    def __init__(self, table: ArrowTable, keys: List[str]):
        """
        Initialize a group by operation.
        
        Args:
            table: ArrowTable
            keys: Column names to group by
        """
        self.table = table
        self.keys = keys
    
    def aggregate(self, aggregations: Dict[str, Callable]) -> ArrowTable:
        """
        Aggregate the grouped data.
        
        Args:
            aggregations: Dictionary mapping column names to aggregation functions
            
        Returns:
            Aggregated ArrowTable
        """
        try:
            import pyarrow.compute as pc
            
            agg_map = {
                "sum": pc.sum,
                "mean": pc.mean,
                "min": pc.min,
                "max": pc.max,
                "count": pc.count,
                "std": pc.stddev,
                "var": pc.variance
            }
            
            agg_exprs = []
            for col, agg_func in aggregations.items():
                if isinstance(agg_func, str):
                    if agg_func not in agg_map:
                        raise ValueError(f"Unknown aggregation function: {agg_func}")
                    agg_exprs.append((col, agg_map[agg_func]))
                else:
                    agg_exprs.append((col, agg_func))
            
            result = pc.group_by(self.keys).aggregate(self.table.table, agg_exprs)
            return ArrowTable(result)
        except (ImportError, AttributeError):
            raise NotImplementedError(
                "Aggregation requires PyArrow with compute module."
            )
    
    def apply(self, func: Callable[[ArrowTable], Any]) -> List[Any]:
        """
        Apply a function to each group.
        
        Args:
            func: Function to apply to each group
            
        Returns:
            List of results
        """
        unique_values = {}
        for key in self.keys:
            col = self.table.column(key)
            unique_values[key] = pa.compute.unique(col).to_pylist()
        
        import itertools
        combinations = list(itertools.product(*[unique_values[key] for key in self.keys]))
        
        results = []
        for combo in combinations:
            filters = []
            for i, key in enumerate(self.keys):
                filters.append(pc.equal(self.table.column(key), combo[i]))
            
            if len(filters) == 1:
                mask = filters[0]
            else:
                mask = filters[0]
                for f in filters[1:]:
                    mask = pc.and_(mask, f)
            
            group = self.table.filter(mask)
            if len(group) > 0:
                results.append(func(group))
        
        return results


def read_parquet(path: str) -> ArrowTable:
    """
    Read a Parquet file into an ArrowTable.
    
    Args:
        path: Path to the Parquet file
        
    Returns:
        ArrowTable
    """
    if not ARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is required for read_parquet(). Install with 'pip install pyarrow'."
        )
    
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return ArrowTable(table)
    except ImportError:
        raise ImportError(
            "PyArrow Parquet is required for read_parquet(). Install with 'pip install pyarrow'."
        )


def write_parquet(table: ArrowTable, path: str, compression: str = "snappy") -> None:
    """
    Write an ArrowTable to a Parquet file.
    
    Args:
        table: ArrowTable
        path: Path to write the Parquet file
        compression: Compression algorithm (snappy, gzip, brotli, zstd, lz4, none)
    """
    if not ARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is required for write_parquet(). Install with 'pip install pyarrow'."
        )
    
    try:
        import pyarrow.parquet as pq
        pq.write_table(table.table, path, compression=compression)
    except ImportError:
        raise ImportError(
            "PyArrow Parquet is required for write_parquet(). Install with 'pip install pyarrow'."
        )


def read_csv(path: str, **kwargs) -> ArrowTable:
    """
    Read a CSV file into an ArrowTable.
    
    Args:
        path: Path to the CSV file
        **kwargs: Additional arguments to pass to pyarrow.csv.read_csv
        
    Returns:
        ArrowTable
    """
    if not ARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is required for read_csv(). Install with 'pip install pyarrow'."
        )
    
    try:
        import pyarrow.csv as csv
        table = csv.read_csv(path, **kwargs)
        return ArrowTable(table)
    except ImportError:
        raise ImportError(
            "PyArrow CSV is required for read_csv(). Install with 'pip install pyarrow'."
        )


def write_csv(table: ArrowTable, path: str, **kwargs) -> None:
    """
    Write an ArrowTable to a CSV file.
    
    Args:
        table: ArrowTable
        path: Path to write the CSV file
        **kwargs: Additional arguments to pass to pyarrow.csv.write_csv
    """
    if not ARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is required for write_csv(). Install with 'pip install pyarrow'."
        )
    
    try:
        import pyarrow.csv as csv
        csv.write_csv(table.table, path, **kwargs)
    except ImportError:
        raise ImportError(
            "PyArrow CSV is required for write_csv(). Install with 'pip install pyarrow'."
        )


def read_json(path: str, **kwargs) -> ArrowTable:
    """
    Read a JSON file into an ArrowTable.
    
    Args:
        path: Path to the JSON file
        **kwargs: Additional arguments to pass to pyarrow.json.read_json
        
    Returns:
        ArrowTable
    """
    if not ARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is required for read_json(). Install with 'pip install pyarrow'."
        )
    
    try:
        import pyarrow.json as json
        table = json.read_json(path, **kwargs)
        return ArrowTable(table)
    except ImportError:
        raise ImportError(
            "PyArrow JSON is required for read_json(). Install with 'pip install pyarrow'."
        )


def tensor_to_arrow(tensor: Tensor) -> pa.Array:
    """
    Convert a Neurenix Tensor to a PyArrow Array.
    
    Args:
        tensor: Neurenix Tensor
        
    Returns:
        PyArrow Array
    """
    if not ARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is required for tensor_to_arrow(). Install with 'pip install pyarrow'."
        )
    
    return pa.array(tensor.numpy())


def arrow_to_tensor(array: Union[pa.Array, pa.ChunkedArray]) -> Tensor:
    """
    Convert a PyArrow Array to a Neurenix Tensor.
    
    Args:
        array: PyArrow Array or ChunkedArray
        
    Returns:
        Neurenix Tensor
    """
    if not ARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is required for arrow_to_tensor(). Install with 'pip install pyarrow'."
        )
    
    if isinstance(array, pa.ChunkedArray):
        return Tensor(array.to_numpy())
    else:
        return Tensor(array.to_numpy())
