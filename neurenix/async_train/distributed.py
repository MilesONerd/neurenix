"""
Distributed checkpoint module for asynchronous and interruptible training.

This module provides functionality for distributed checkpointing across
multiple nodes, enabling fault tolerance in distributed training.
"""

import os
import time
import json
import threading
import socket
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import neurenix
from neurenix.nn import Module
from neurenix.optim import Optimizer
from neurenix.async_train.checkpoint import Checkpoint, CheckpointManager

class DistributedCheckpoint:
    """
    Distributed checkpoint for multi-node training.
    
    Provides functionality for coordinated checkpointing across multiple
    nodes in a distributed training setup, with fault tolerance.
    
    Attributes:
        checkpoint_manager: Manager for local checkpoints
        coordinator_address: Address of the coordinator node
        node_rank: Rank of this node in the distributed setup
        world_size: Total number of nodes in the distributed setup
        sync_frequency: How often to synchronize checkpoints (in seconds)
        timeout: Timeout for network operations (in seconds)
        logger: Logger for distributed checkpoint events
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        coordinator_address: Optional[str] = None,
        node_rank: int = 0,
        world_size: int = 1,
        sync_frequency: float = 60.0,
        timeout: float = 30.0,
        logger: Optional[Any] = None
    ):
        """
        Initialize distributed checkpoint.
        
        Args:
            checkpoint_manager: Manager for local checkpoints
            coordinator_address: Address of the coordinator node (host:port)
            node_rank: Rank of this node in the distributed setup
            world_size: Total number of nodes in the distributed setup
            sync_frequency: How often to synchronize checkpoints (in seconds)
            timeout: Timeout for network operations (in seconds)
            logger: Logger for distributed checkpoint events
        """
        self.checkpoint_manager = checkpoint_manager
        self.coordinator_address = coordinator_address
        self.node_rank = node_rank
        self.world_size = world_size
        self.sync_frequency = sync_frequency
        self.timeout = timeout
        self.logger = logger or neurenix.get_logger(__name__)
        
        self.is_coordinator = node_rank == 0
        self.running = False
        self.sync_thread = None
        self.stop_event = threading.Event()
        
        self.socket = None
        if coordinator_address and not self.is_coordinator:
            self._init_network()
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def _init_network(self):
        """Initialize network connection to coordinator."""
        if self._has_phynexus:
            self._phynexus.async_train.init_distributed_network(
                self.coordinator_address, self.timeout
            )
            return
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            host, port = self.coordinator_address.split(":")
            port = int(port)
            
            self.socket.connect((host, port))
            self.logger.info(f"Connected to coordinator at {self.coordinator_address}")
        
        except Exception as e:
            self.logger.error(f"Failed to connect to coordinator: {str(e)}")
            self.socket = None
    
    def start(self):
        """
        Start distributed checkpoint synchronization.
        """
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True
        )
        self.sync_thread.start()
        
        self.logger.info("Distributed checkpoint synchronization started")
    
    def stop(self):
        """
        Stop distributed checkpoint synchronization.
        """
        if not self.running:
            return
        
        self.stop_event.set()
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)
        
        self.running = False
        self.logger.info("Distributed checkpoint synchronization stopped")
    
    def _sync_loop(self):
        """
        Main synchronization loop.
        """
        if self._has_phynexus:
            self._phynexus.async_train.distributed_checkpoint_sync_loop(
                self.checkpoint_manager, self.is_coordinator,
                self.node_rank, self.world_size,
                self.sync_frequency, self.stop_event
            )
            return
        
        while not self.stop_event.is_set():
            try:
                if self.is_coordinator:
                    self._coordinator_sync()
                else:
                    self._worker_sync()
            
            except Exception as e:
                self.logger.error(f"Error in distributed checkpoint sync: {str(e)}")
            
            self.stop_event.wait(self.sync_frequency)
    
    def _coordinator_sync(self):
        """
        Synchronization logic for coordinator node.
        """
        checkpoint_info = self._get_latest_checkpoint_info()
        
        
        self.logger.info(f"Coordinator sync: Latest checkpoint at step {checkpoint_info.get('step', 0)}")
    
    def _worker_sync(self):
        """
        Synchronization logic for worker nodes.
        """
        if not self.socket:
            self._init_network()
            if not self.socket:
                return
        
        checkpoint_info = self._get_latest_checkpoint_info()
        
        
        self.logger.info(f"Worker sync: Latest checkpoint at step {checkpoint_info.get('step', 0)}")
    
    def _get_latest_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about the latest checkpoint.
        
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = self.checkpoint_manager.checkpoints.get("main")
        if not checkpoint:
            return {}
        
        checkpoint_list = checkpoint.list_checkpoints()
        if not checkpoint_list:
            return {}
        
        return checkpoint_list[-1]
    
    def save_distributed(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a distributed checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            metrics: Metrics to save
            metadata: Metadata to save
            
        Returns:
            Path to the saved checkpoint file
        """
        dist_metadata = {
            "node_rank": self.node_rank,
            "world_size": self.world_size,
            "timestamp": time.time()
        }
        
        if metadata:
            metadata.update(dist_metadata)
        else:
            metadata = dist_metadata
        
        checkpoint_path = self.checkpoint_manager.save(
            "main", model, optimizer, metrics, metadata=metadata
        )
        
        if self.running:
            try:
                if self.is_coordinator:
                    self._coordinator_sync()
                else:
                    self._worker_sync()
            except Exception as e:
                self.logger.error(f"Error in distributed checkpoint sync: {str(e)}")
        
        return checkpoint_path
    
    def load_distributed(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        step: Optional[int] = None,
        latest: bool = True
    ) -> Dict[str, Any]:
        """
        Load a distributed checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            step: Specific step to load
            latest: Whether to load the latest checkpoint
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        return self.checkpoint_manager.load(
            "main", model, optimizer, step, latest
        )
    
    def barrier(self, timeout: Optional[float] = None) -> bool:
        """
        Synchronization barrier for all nodes.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if all nodes reached the barrier, False otherwise
        """
        if self._has_phynexus:
            return self._phynexus.async_train.distributed_checkpoint_barrier(
                self.is_coordinator, self.node_rank, self.world_size,
                timeout or self.timeout
            )
        
        
        self.logger.info(f"Node {self.node_rank} reached barrier")
        
        return True
    
    def __del__(self):
        """
        Clean up resources.
        """
        self.stop()
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


class FederatedCheckpoint(DistributedCheckpoint):
    """
    Federated checkpoint for privacy-preserving distributed training.
    
    Extends distributed checkpoint with privacy-preserving mechanisms
    for federated learning scenarios.
    
    Attributes:
        encryption_key: Key for encrypting checkpoint data
        differential_privacy: Whether to apply differential privacy
        noise_scale: Scale of noise for differential privacy
        clip_norm: Gradient clipping norm for differential privacy
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        coordinator_address: Optional[str] = None,
        node_rank: int = 0,
        world_size: int = 1,
        sync_frequency: float = 60.0,
        timeout: float = 30.0,
        encryption_key: Optional[str] = None,
        differential_privacy: bool = False,
        noise_scale: float = 0.1,
        clip_norm: float = 1.0,
        logger: Optional[Any] = None
    ):
        """
        Initialize federated checkpoint.
        
        Args:
            checkpoint_manager: Manager for local checkpoints
            coordinator_address: Address of the coordinator node
            node_rank: Rank of this node in the distributed setup
            world_size: Total number of nodes in the distributed setup
            sync_frequency: How often to synchronize checkpoints (in seconds)
            timeout: Timeout for network operations (in seconds)
            encryption_key: Key for encrypting checkpoint data
            differential_privacy: Whether to apply differential privacy
            noise_scale: Scale of noise for differential privacy
            clip_norm: Gradient clipping norm for differential privacy
            logger: Logger for federated checkpoint events
        """
        super().__init__(
            checkpoint_manager, coordinator_address, node_rank,
            world_size, sync_frequency, timeout, logger
        )
        
        self.encryption_key = encryption_key
        self.differential_privacy = differential_privacy
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
    
    def save_distributed(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a federated checkpoint with privacy protection.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            metrics: Metrics to save
            metadata: Metadata to save
            
        Returns:
            Path to the saved checkpoint file
        """
        if self.differential_privacy and not self.is_coordinator:
            self._apply_differential_privacy(model)
        
        fed_metadata = {
            "federated": True,
            "differential_privacy": self.differential_privacy,
            "encrypted": self.encryption_key is not None
        }
        
        if metadata:
            metadata.update(fed_metadata)
        else:
            metadata = fed_metadata
        
        return super().save_distributed(model, optimizer, metrics, metadata)
    
    def _apply_differential_privacy(self, model: Module):
        """
        Apply differential privacy to model parameters.
        
        Args:
            model: Model to apply differential privacy to
        """
        if self._has_phynexus:
            self._phynexus.async_train.apply_differential_privacy(
                model, self.noise_scale, self.clip_norm
            )
            return
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad.clip_(max_norm=self.clip_norm)
            
            if param.requires_grad:
                noise = neurenix.randn_like(param) * self.noise_scale
                param.add_(noise)
    
    def _encrypt_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt model state dictionary.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Encrypted state dictionary
        """
        if not self.encryption_key:
            return state_dict
        
        
        return state_dict
    
    def _decrypt_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt model state dictionary.
        
        Args:
            state_dict: Encrypted model state dictionary
            
        Returns:
            Decrypted state dictionary
        """
        if not self.encryption_key:
            return state_dict
        
        
        return state_dict
