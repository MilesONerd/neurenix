"""
Asynchronous trainer module for interruptible training.

This module provides functionality for asynchronous training with continuous
checkpointing and automatic resume, even in unstable environments.
"""

import os
import time
import threading
import queue
import signal
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import neurenix
from neurenix.nn import Module
from neurenix.optim import Optimizer
from neurenix.async_train.checkpoint import Checkpoint, CheckpointManager

class AsyncTrainer:
    """
    Asynchronous trainer for interruptible training.
    
    Provides asynchronous training with continuous checkpointing and
    automatic resume capabilities, designed for unstable environments.
    
    Attributes:
        model: Neural network model
        optimizer: Optimizer for training
        loss_fn: Loss function
        checkpoint_manager: Manager for checkpoints
        checkpoint_frequency: How often to save checkpoints (in seconds or steps)
        frequency_unit: Unit for checkpoint_frequency ('seconds' or 'steps')
        max_retries: Maximum number of retries for failed operations
        retry_delay: Delay between retries (in seconds)
        logger: Logger for training events
    """
    
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        checkpoint_dir: str,
        checkpoint_frequency: Union[int, float] = 60,
        frequency_unit: str = "seconds",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize asynchronous trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimizer for training
            loss_fn: Loss function
            checkpoint_dir: Directory to save checkpoints
            checkpoint_frequency: How often to save checkpoints (in seconds or steps)
            frequency_unit: Unit for checkpoint_frequency ('seconds' or 'steps')
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries (in seconds)
            logger: Logger for training events
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_frequency = checkpoint_frequency
        self.frequency_unit = frequency_unit
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger or logging.getLogger(__name__)
        
        self.checkpoint_manager = CheckpointManager(auto_save=True, auto_load=True)
        self.checkpoint_manager.create_checkpoint(
            name="main",
            directory=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            save_frequency=checkpoint_frequency,
            frequency_unit=frequency_unit
        )
        
        self.step = 0
        self.epoch = 0
        self.best_metrics = {}
        self.training_history = []
        self.is_training = False
        self.stop_requested = False
        
        self.training_thread = None
        self.checkpoint_thread = None
        self.event_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def _signal_handler(self, sig, frame):
        """
        Handle signals (SIGINT, SIGTERM) to gracefully stop training.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        if sig == signal.SIGINT:
            self.logger.info("Received SIGINT, stopping training gracefully...")
        elif sig == signal.SIGTERM:
            self.logger.info("Received SIGTERM, stopping training gracefully...")
        
        self.stop_requested = True
        
        if not self.is_training:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            if sig == signal.SIGINT:
                self._original_sigint_handler(sig, frame)
            elif sig == signal.SIGTERM:
                self._original_sigterm_handler(sig, frame)
    
    def train(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 1,
        callbacks: Optional[List[Callable]] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        async_mode: bool = True
    ):
        """
        Train the model with automatic checkpointing and resume.
        
        Args:
            train_loader: Data loader for training
            val_loader: Data loader for validation
            epochs: Number of epochs to train
            callbacks: List of callback functions
            metrics: Dictionary of metric functions
            async_mode: Whether to train asynchronously
            
        Returns:
            Training history
        """
        if self._has_phynexus:
            return self._phynexus.async_train.train(
                self.model, self.optimizer, self.loss_fn,
                train_loader, val_loader, epochs,
                callbacks, metrics, async_mode,
                self.checkpoint_manager
            )
        
        if async_mode:
            return self._train_async(train_loader, val_loader, epochs, callbacks, metrics)
        else:
            return self._train_sync(train_loader, val_loader, epochs, callbacks, metrics)
    
    def _train_sync(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 1,
        callbacks: Optional[List[Callable]] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """
        Train the model synchronously with automatic checkpointing and resume.
        
        Args:
            train_loader: Data loader for training
            val_loader: Data loader for validation
            epochs: Number of epochs to train
            callbacks: List of callback functions
            metrics: Dictionary of metric functions
            
        Returns:
            Training history
        """
        self.is_training = True
        self.stop_requested = False
        
        callbacks = callbacks or []
        metrics = metrics or {}
        
        checkpoint_data = self.checkpoint_manager.load("main", self.model, self.optimizer)
        if checkpoint_data and "step" in checkpoint_data:
            self.step = checkpoint_data["step"]
            if "epoch" in checkpoint_data:
                self.epoch = checkpoint_data["epoch"]
            if "metrics" in checkpoint_data:
                self.best_metrics = checkpoint_data["metrics"]
            if "history" in checkpoint_data:
                self.training_history = checkpoint_data["history"]
            
            self.logger.info(f"Resuming training from step {self.step}, epoch {self.epoch}")
        
        try:
            for epoch in range(self.epoch, epochs):
                self.epoch = epoch
                
                if self.stop_requested:
                    self.logger.info("Training stopped by user")
                    break
                
                self.model.train()
                train_metrics = self._run_epoch(train_loader, is_training=True, metrics=metrics)
                
                val_metrics = {}
                if val_loader is not None:
                    self.model.eval()
                    val_metrics = self._run_epoch(val_loader, is_training=False, metrics=metrics)
                
                epoch_metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
                
                self.training_history.append({
                    "epoch": epoch,
                    "step": self.step,
                    **epoch_metrics
                })
                
                for callback in callbacks:
                    callback(self.model, epoch, epoch_metrics)
                
                self.checkpoint_manager.save(
                    "main",
                    self.model,
                    self.optimizer,
                    metrics=self.best_metrics,
                    metadata={
                        "epoch": epoch,
                        "step": self.step,
                        "history": self.training_history
                    }
                )
                
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
                self.logger.info(f"Epoch {epoch}/{epochs-1}, {metrics_str}")
        
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.checkpoint_manager.save(
                "main",
                self.model,
                self.optimizer,
                metrics=self.best_metrics,
                metadata={
                    "epoch": self.epoch,
                    "step": self.step,
                    "history": self.training_history,
                    "error": str(e)
                }
            )
            raise
        
        finally:
            self.is_training = False
        
        return self.training_history
    
    def _run_epoch(
        self,
        data_loader: Any,
        is_training: bool = True,
        metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, float]:
        """
        Run a single epoch of training or validation.
        
        Args:
            data_loader: Data loader
            is_training: Whether this is a training epoch
            metrics: Dictionary of metric functions
            
        Returns:
            Dictionary of metrics
        """
        metrics = metrics or {}
        metric_values = {name: 0.0 for name in metrics}
        total_loss = 0.0
        num_batches = 0
        
        context = neurenix.enable_grad if is_training else neurenix.no_grad
        
        with context():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if self.stop_requested:
                    break
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    self.step += 1
                    
                    if (self.frequency_unit == "steps" and 
                        self.step % self.checkpoint_frequency == 0):
                        self.checkpoint_manager.save(
                            "main",
                            self.model,
                            self.optimizer,
                            metrics=self.best_metrics,
                            metadata={
                                "epoch": self.epoch,
                                "step": self.step,
                                "history": self.training_history
                            }
                        )
                
                total_loss += loss.item()
                for name, metric_fn in metrics.items():
                    metric_values[name] += metric_fn(outputs, targets).item()
                
                num_batches += 1
        
        avg_metrics = {"loss": total_loss / max(1, num_batches)}
        for name in metrics:
            avg_metrics[name] = metric_values[name] / max(1, num_batches)
        
        return avg_metrics
    
    def _train_async(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 1,
        callbacks: Optional[List[Callable]] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """
        Train the model asynchronously with automatic checkpointing and resume.
        
        Args:
            train_loader: Data loader for training
            val_loader: Data loader for validation
            epochs: Number of epochs to train
            callbacks: List of callback functions
            metrics: Dictionary of metric functions
            
        Returns:
            Training history
        """
        if self.is_training:
            self.logger.warning("Training already in progress")
            return None
        
        self.is_training = True
        self.stop_requested = False
        
        callbacks = callbacks or []
        metrics = metrics or {}
        
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(train_loader, val_loader, epochs, callbacks, metrics)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        
        self.checkpoint_thread = threading.Thread(
            target=self._checkpoint_worker
        )
        self.checkpoint_thread.daemon = True
        self.checkpoint_thread.start()
        
        return self.training_thread
    
    def _training_worker(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 1,
        callbacks: Optional[List[Callable]] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """
        Worker function for asynchronous training.
        
        Args:
            train_loader: Data loader for training
            val_loader: Data loader for validation
            epochs: Number of epochs to train
            callbacks: List of callback functions
            metrics: Dictionary of metric functions
        """
        try:
            checkpoint_data = self.checkpoint_manager.load("main", self.model, self.optimizer)
            if checkpoint_data and "step" in checkpoint_data:
                self.step = checkpoint_data["step"]
                if "epoch" in checkpoint_data:
                    self.epoch = checkpoint_data["epoch"]
                if "metrics" in checkpoint_data:
                    self.best_metrics = checkpoint_data["metrics"]
                if "history" in checkpoint_data:
                    self.training_history = checkpoint_data["history"]
                
                self.logger.info(f"Resuming training from step {self.step}, epoch {self.epoch}")
            
            for epoch in range(self.epoch, epochs):
                self.epoch = epoch
                
                if self.stop_requested:
                    self.logger.info("Training stopped by user")
                    self.event_queue.put(("stop", None))
                    break
                
                self.model.train()
                train_metrics = self._run_epoch(train_loader, is_training=True, metrics=metrics)
                
                val_metrics = {}
                if val_loader is not None:
                    self.model.eval()
                    val_metrics = self._run_epoch(val_loader, is_training=False, metrics=metrics)
                
                epoch_metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
                
                self.training_history.append({
                    "epoch": epoch,
                    "step": self.step,
                    **epoch_metrics
                })
                
                for callback in callbacks:
                    callback(self.model, epoch, epoch_metrics)
                
                self.event_queue.put(("epoch_end", {
                    "epoch": epoch,
                    "metrics": epoch_metrics
                }))
                
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
                self.logger.info(f"Epoch {epoch}/{epochs-1}, {metrics_str}")
            
            self.event_queue.put(("training_complete", {
                "epochs": epochs,
                "history": self.training_history
            }))
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.event_queue.put(("error", {
                "error": str(e),
                "epoch": self.epoch,
                "step": self.step
            }))
            self.result_queue.put(("error", str(e)))
            
        finally:
            self.is_training = False
    
    def _checkpoint_worker(self):
        """
        Worker function for asynchronous checkpointing.
        """
        last_checkpoint_time = time.time()
        
        while True:
            try:
                try:
                    event_type, event_data = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    event_type, event_data = None, None
                
                if event_type == "stop" or event_type == "training_complete":
                    self.checkpoint_manager.save(
                        "main",
                        self.model,
                        self.optimizer,
                        metrics=self.best_metrics,
                        metadata={
                            "epoch": self.epoch,
                            "step": self.step,
                            "history": self.training_history,
                            "status": "completed" if event_type == "training_complete" else "stopped"
                        }
                    )
                    
                    if event_type == "training_complete":
                        self.result_queue.put(("complete", self.training_history))
                    else:
                        self.result_queue.put(("stopped", self.training_history))
                    
                    break
                
                elif event_type == "error":
                    self.checkpoint_manager.save(
                        "main",
                        self.model,
                        self.optimizer,
                        metrics=self.best_metrics,
                        metadata={
                            "epoch": self.epoch,
                            "step": self.step,
                            "history": self.training_history,
                            "error": event_data["error"],
                            "status": "error"
                        }
                    )
                    
                    break
                
                elif event_type == "epoch_end":
                    self.checkpoint_manager.save(
                        "main",
                        self.model,
                        self.optimizer,
                        metrics=self.best_metrics,
                        metadata={
                            "epoch": event_data["epoch"],
                            "step": self.step,
                            "history": self.training_history
                        }
                    )
                    
                    last_checkpoint_time = time.time()
                
                current_time = time.time()
                if (self.frequency_unit == "seconds" and 
                    current_time - last_checkpoint_time >= self.checkpoint_frequency):
                    self.checkpoint_manager.save(
                        "main",
                        self.model,
                        self.optimizer,
                        metrics=self.best_metrics,
                        metadata={
                            "epoch": self.epoch,
                            "step": self.step,
                            "history": self.training_history
                        }
                    )
                    
                    last_checkpoint_time = current_time
            
            except Exception as e:
                self.logger.error(f"Error in checkpoint thread: {str(e)}")
                try:
                    self.checkpoint_manager.save(
                        "main",
                        self.model,
                        self.optimizer,
                        metrics=self.best_metrics,
                        metadata={
                            "epoch": self.epoch,
                            "step": self.step,
                            "history": self.training_history,
                            "error": str(e),
                            "status": "checkpoint_error"
                        }
                    )
                except Exception as save_error:
                    self.logger.error(f"Failed to save checkpoint: {str(save_error)}")
    
    def wait(self, timeout: Optional[float] = None) -> Tuple[str, Any]:
        """
        Wait for training to complete.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (status, result)
        """
        if not self.is_training:
            return ("not_running", None)
        
        try:
            status, result = self.result_queue.get(timeout=timeout)
            return (status, result)
        except queue.Empty:
            return ("running", None)
    
    def stop(self):
        """
        Stop training gracefully.
        """
        if not self.is_training:
            return
        
        self.logger.info("Stopping training gracefully...")
        self.stop_requested = True
    
    def __del__(self):
        """
        Clean up resources.
        """
        try:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
        except (AttributeError, ValueError):
            pass
