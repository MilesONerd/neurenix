"""
Checkpoint module for asynchronous and interruptible training.

This module provides functionality for saving and loading model checkpoints
during training, enabling continuous checkpointing and automatic resume.
"""

import os
import json
import time
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import neurenix
from neurenix.nn import Module
from neurenix.optim import Optimizer

class Checkpoint:
    """
    Checkpoint for saving and loading model state.
    
    Attributes:
        directory: Directory to save checkpoints
        prefix: Prefix for checkpoint files
        max_to_keep: Maximum number of checkpoints to keep
        save_optimizer: Whether to save optimizer state
        save_metrics: Whether to save training metrics
        save_frequency: How often to save checkpoints (in seconds or steps)
        frequency_unit: Unit for save_frequency ('seconds' or 'steps')
    """
    
    def __init__(
        self,
        directory: str,
        prefix: str = "checkpoint",
        max_to_keep: int = 5,
        save_optimizer: bool = True,
        save_metrics: bool = True,
        save_frequency: Union[int, float] = 300,
        frequency_unit: str = "seconds",
        use_atomic_write: bool = True
    ):
        """
        Initialize checkpoint.
        
        Args:
            directory: Directory to save checkpoints
            prefix: Prefix for checkpoint files
            max_to_keep: Maximum number of checkpoints to keep
            save_optimizer: Whether to save optimizer state
            save_metrics: Whether to save training metrics
            save_frequency: How often to save checkpoints (in seconds or steps)
            frequency_unit: Unit for save_frequency ('seconds' or 'steps')
            use_atomic_write: Whether to use atomic write operations
        """
        self.directory = directory
        self.prefix = prefix
        self.max_to_keep = max_to_keep
        self.save_optimizer = save_optimizer
        self.save_metrics = save_metrics
        self.save_frequency = save_frequency
        self.frequency_unit = frequency_unit
        self.use_atomic_write = use_atomic_write
        
        os.makedirs(directory, exist_ok=True)
        
        self.last_save_time = time.time()
        self.step_counter = 0
        self.checkpoint_files: List[str] = []
        
        self._load_checkpoint_files()
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def _load_checkpoint_files(self):
        """Load existing checkpoint files from directory."""
        if os.path.exists(self.directory):
            files = [f for f in os.listdir(self.directory) 
                    if f.startswith(self.prefix) and f.endswith(".pt")]
            
            files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
            
            self.checkpoint_files = files
    
    def should_save(self) -> bool:
        """
        Check if a checkpoint should be saved.
        
        Returns:
            True if a checkpoint should be saved, False otherwise
        """
        if self.frequency_unit == "seconds":
            current_time = time.time()
            return (current_time - self.last_save_time) >= self.save_frequency
        else:  # steps
            return self.step_counter % self.save_frequency == 0
    
    def save(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save (optional)
            metrics: Training metrics to save (optional)
            step: Current training step (optional)
            metadata: Additional metadata to save (optional)
            
        Returns:
            Path to the saved checkpoint file
        """
        if self._has_phynexus:
            checkpoint_path = self._phynexus.async_train.save_checkpoint(
                self.directory, self.prefix, model, optimizer, metrics, step, metadata,
                self.use_atomic_write
            )
            
            self.last_save_time = time.time()
            self.step_counter += 1
            
            self._add_checkpoint_file(checkpoint_path)
            
            return checkpoint_path
        
        if step is None:
            step = self.step_counter
        
        checkpoint_file = f"{self.prefix}_{step}.pt"
        checkpoint_path = os.path.join(self.directory, checkpoint_file)
        
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "step": step,
            "timestamp": time.time()
        }
        
        if self.save_optimizer and optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        
        if self.save_metrics and metrics is not None:
            checkpoint_data["metrics"] = metrics
        
        if metadata is not None:
            checkpoint_data["metadata"] = metadata
        
        if self.use_atomic_write:
            with tempfile.NamedTemporaryFile(delete=False, dir=self.directory) as tmp:
                neurenix.save(checkpoint_data, tmp.name)
                tmp_name = tmp.name
            
            shutil.move(tmp_name, checkpoint_path)
        else:
            neurenix.save(checkpoint_data, checkpoint_path)
        
        self.last_save_time = time.time()
        self.step_counter += 1
        
        self._add_checkpoint_file(checkpoint_file)
        
        return checkpoint_path
    
    def _add_checkpoint_file(self, checkpoint_file: str):
        """
        Add a checkpoint file to tracking and manage max_to_keep.
        
        Args:
            checkpoint_file: Checkpoint file to add
        """
        if os.path.dirname(checkpoint_file):
            checkpoint_file = os.path.basename(checkpoint_file)
        
        if checkpoint_file not in self.checkpoint_files:
            self.checkpoint_files.append(checkpoint_file)
        
        while len(self.checkpoint_files) > self.max_to_keep:
            oldest_file = self.checkpoint_files.pop(0)
            oldest_path = os.path.join(self.directory, oldest_file)
            if os.path.exists(oldest_path):
                os.remove(oldest_path)
    
    def load(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        step: Optional[int] = None,
        latest: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            step: Specific step to load (optional)
            latest: Whether to load the latest checkpoint (ignored if step is provided)
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        if self._has_phynexus:
            return self._phynexus.async_train.load_checkpoint(
                self.directory, self.prefix, model, optimizer, step, latest
            )
        
        checkpoint_file = self._find_checkpoint_file(step, latest)
        if checkpoint_file is None:
            raise FileNotFoundError(f"No checkpoint found in {self.directory}")
        
        checkpoint_path = os.path.join(self.directory, checkpoint_file)
        
        checkpoint_data = neurenix.load(checkpoint_path)
        
        model.load_state_dict(checkpoint_data["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
        if "step" in checkpoint_data:
            self.step_counter = checkpoint_data["step"]
        
        result = {k: v for k, v in checkpoint_data.items() 
                 if k not in ["model_state_dict", "optimizer_state_dict"]}
        
        return result
    
    def _find_checkpoint_file(
        self, 
        step: Optional[int] = None, 
        latest: bool = True
    ) -> Optional[str]:
        """
        Find a checkpoint file to load.
        
        Args:
            step: Specific step to load
            latest: Whether to load the latest checkpoint
            
        Returns:
            Checkpoint filename or None if not found
        """
        if not self.checkpoint_files:
            return None
        
        if step is not None:
            for file in self.checkpoint_files:
                file_step = int(file.split("_")[-1].split(".")[0])
                if file_step == step:
                    return file
            return None
        
        if latest:
            return self.checkpoint_files[-1]
        
        return self.checkpoint_files[0]
    
    def get_latest_step(self) -> Optional[int]:
        """
        Get the latest checkpoint step.
        
        Returns:
            Latest step number or None if no checkpoints exist
        """
        if not self.checkpoint_files:
            return None
        
        latest_file = self.checkpoint_files[-1]
        return int(latest_file.split("_")[-1].split(".")[0])
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of dictionaries containing checkpoint information
        """
        result = []
        
        for file in self.checkpoint_files:
            checkpoint_path = os.path.join(self.directory, file)
            try:
                checkpoint_data = neurenix.load(checkpoint_path, map_location="cpu")
                
                info = {
                    "filename": file,
                    "step": checkpoint_data.get("step", 0),
                    "timestamp": checkpoint_data.get("timestamp", 0)
                }
                
                if "metrics" in checkpoint_data:
                    info["metrics"] = checkpoint_data["metrics"]
                
                if "metadata" in checkpoint_data:
                    info["metadata"] = checkpoint_data["metadata"]
                
                result.append(info)
            except Exception as e:
                print(f"Warning: Could not load checkpoint {file}: {str(e)}")
        
        return result


class CheckpointManager:
    """
    Manager for multiple checkpoints.
    
    Attributes:
        checkpoints: Dictionary mapping names to Checkpoint objects
        auto_save: Whether to automatically save checkpoints
        auto_load: Whether to automatically load checkpoints on creation
    """
    
    def __init__(
        self,
        auto_save: bool = True,
        auto_load: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            auto_save: Whether to automatically save checkpoints
            auto_load: Whether to automatically load checkpoints on creation
        """
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.auto_save = auto_save
        self.auto_load = auto_load
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def register_checkpoint(
        self,
        name: str,
        checkpoint: Checkpoint,
        model: Optional[Module] = None,
        optimizer: Optional[Optimizer] = None
    ):
        """
        Register a checkpoint.
        
        Args:
            name: Name for the checkpoint
            checkpoint: Checkpoint object
            model: Model to associate with the checkpoint
            optimizer: Optimizer to associate with the checkpoint
        """
        self.checkpoints[name] = checkpoint
        
        if self.auto_load and model is not None:
            try:
                checkpoint.load(model, optimizer, latest=True)
            except FileNotFoundError:
                pass
    
    def create_checkpoint(
        self,
        name: str,
        directory: str,
        model: Optional[Module] = None,
        optimizer: Optional[Optimizer] = None,
        **kwargs
    ) -> Checkpoint:
        """
        Create and register a new checkpoint.
        
        Args:
            name: Name for the checkpoint
            directory: Directory to save checkpoints
            model: Model to associate with the checkpoint
            optimizer: Optimizer to associate with the checkpoint
            **kwargs: Additional arguments for Checkpoint constructor
            
        Returns:
            Created Checkpoint object
        """
        checkpoint = Checkpoint(directory=directory, **kwargs)
        self.register_checkpoint(name, checkpoint, model, optimizer)
        return checkpoint
    
    def save_all(
        self,
        models: Optional[Dict[str, Module]] = None,
        optimizers: Optional[Dict[str, Optimizer]] = None,
        metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Save all checkpoints.
        
        Args:
            models: Dictionary mapping names to models
            optimizers: Dictionary mapping names to optimizers
            metrics: Dictionary mapping names to metrics
            metadata: Dictionary mapping names to metadata
            
        Returns:
            Dictionary mapping names to saved checkpoint paths
        """
        result = {}
        
        for name, checkpoint in self.checkpoints.items():
            model = None if models is None else models.get(name)
            optimizer = None if optimizers is None else optimizers.get(name)
            metric = None if metrics is None else metrics.get(name)
            meta = None if metadata is None else metadata.get(name)
            
            if model is not None:
                path = checkpoint.save(model, optimizer, metric, metadata=meta)
                result[name] = path
        
        return result
    
    def save(
        self,
        name: str,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a specific checkpoint.
        
        Args:
            name: Name of the checkpoint
            model: Model to save
            optimizer: Optimizer to save
            metrics: Metrics to save
            metadata: Metadata to save
            
        Returns:
            Path to the saved checkpoint file
        """
        if name not in self.checkpoints:
            raise KeyError(f"Checkpoint '{name}' not found")
        
        return self.checkpoints[name].save(model, optimizer, metrics, metadata=metadata)
    
    def load(
        self,
        name: str,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        step: Optional[int] = None,
        latest: bool = True
    ) -> Dict[str, Any]:
        """
        Load a specific checkpoint.
        
        Args:
            name: Name of the checkpoint
            model: Model to load state into
            optimizer: Optimizer to load state into
            step: Specific step to load
            latest: Whether to load the latest checkpoint
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        if name not in self.checkpoints:
            raise KeyError(f"Checkpoint '{name}' not found")
        
        return self.checkpoints[name].load(model, optimizer, step, latest)
    
    def check_and_save(
        self,
        models: Optional[Dict[str, Module]] = None,
        optimizers: Optional[Dict[str, Optimizer]] = None,
        metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Check if checkpoints should be saved and save them if needed.
        
        Args:
            models: Dictionary mapping names to models
            optimizers: Dictionary mapping names to optimizers
            metrics: Dictionary mapping names to metrics
            metadata: Dictionary mapping names to metadata
            
        Returns:
            Dictionary mapping names to saved checkpoint paths
        """
        result = {}
        
        for name, checkpoint in self.checkpoints.items():
            if checkpoint.should_save():
                model = None if models is None else models.get(name)
                optimizer = None if optimizers is None else optimizers.get(name)
                metric = None if metrics is None else metrics.get(name)
                meta = None if metadata is None else metadata.get(name)
                
                if model is not None:
                    path = checkpoint.save(model, optimizer, metric, metadata=meta)
                    result[name] = path
        
        return result
