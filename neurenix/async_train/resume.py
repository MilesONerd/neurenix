"""
Automatic resume module for asynchronous and interruptible training.

This module provides functionality for automatically resuming training
after interruptions, designed for unstable environments.
"""

import os
import time
import logging
import threading
import signal
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import neurenix
from neurenix.nn import Module
from neurenix.optim import Optimizer
from neurenix.async_train.checkpoint import Checkpoint, CheckpointManager

class AutoResume:
    """
    Automatic resume for interrupted training.
    
    Provides functionality to automatically detect and recover from
    training interruptions, designed for unstable environments.
    
    Attributes:
        checkpoint_manager: Manager for checkpoints
        model: Neural network model
        optimizer: Optimizer for training
        max_retries: Maximum number of retries for failed operations
        retry_delay: Delay between retries (in seconds)
        logger: Logger for training events
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        model: Module,
        optimizer: Optimizer,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize automatic resume.
        
        Args:
            checkpoint_manager: Manager for checkpoints
            model: Neural network model
            optimizer: Optimizer for training
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries (in seconds)
            logger: Logger for training events
        """
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.optimizer = optimizer
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger or logging.getLogger(__name__)
        
        self.is_resuming = False
        self.resume_attempts = 0
        self.last_checkpoint_time = time.time()
        self.last_checkpoint_step = 0
        self.last_checkpoint_epoch = 0
        
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
            self.logger.info("Received SIGINT, saving checkpoint before exit...")
        elif sig == signal.SIGTERM:
            self.logger.info("Received SIGTERM, saving checkpoint before exit...")
        
        try:
            self.checkpoint_manager.save(
                "main",
                self.model,
                self.optimizer,
                metadata={
                    "interrupted": True,
                    "signal": sig,
                    "time": time.time()
                }
            )
            self.logger.info("Checkpoint saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
        
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        signal.signal(signal.SIGTERM, self._original_sigterm_handler)
        if sig == signal.SIGINT:
            self._original_sigint_handler(sig, frame)
        elif sig == signal.SIGTERM:
            self._original_sigterm_handler(sig, frame)
    
    def check_for_interruption(self) -> bool:
        """
        Check if training was previously interrupted.
        
        Returns:
            True if training was interrupted, False otherwise
        """
        if self._has_phynexus:
            return self._phynexus.async_train.check_for_interruption(
                self.checkpoint_manager
            )
        
        try:
            checkpoints = self.checkpoint_manager.checkpoints.get("main")
            if not checkpoints:
                return False
            
            checkpoint_list = checkpoints.list_checkpoints()
            if not checkpoint_list:
                return False
            
            latest_checkpoint = checkpoint_list[-1]
            
            metadata = latest_checkpoint.get("metadata", {})
            return metadata.get("interrupted", False)
        
        except Exception as e:
            self.logger.error(f"Error checking for interruption: {str(e)}")
            return False
    
    def resume(self) -> Dict[str, Any]:
        """
        Resume training from the latest checkpoint.
        
        Returns:
            Dictionary containing checkpoint metadata
        """
        if self._has_phynexus:
            return self._phynexus.async_train.resume_training(
                self.checkpoint_manager, self.model, self.optimizer,
                self.max_retries, self.retry_delay
            )
        
        self.is_resuming = True
        self.resume_attempts = 0
        
        while self.resume_attempts < self.max_retries:
            try:
                checkpoint_data = self.checkpoint_manager.load(
                    "main", self.model, self.optimizer, latest=True
                )
                
                if "step" in checkpoint_data:
                    self.last_checkpoint_step = checkpoint_data["step"]
                if "epoch" in checkpoint_data.get("metadata", {}):
                    self.last_checkpoint_epoch = checkpoint_data["metadata"]["epoch"]
                
                self.last_checkpoint_time = time.time()
                self.is_resuming = False
                
                self.logger.info(f"Successfully resumed from checkpoint at step {self.last_checkpoint_step}, epoch {self.last_checkpoint_epoch}")
                
                return checkpoint_data
            
            except Exception as e:
                self.resume_attempts += 1
                self.logger.error(f"Failed to resume (attempt {self.resume_attempts}/{self.max_retries}): {str(e)}")
                
                if self.resume_attempts < self.max_retries:
                    time.sleep(self.retry_delay)
        
        self.is_resuming = False
        self.logger.error(f"Failed to resume after {self.max_retries} attempts")
        return {}
    
    def auto_resume_if_needed(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Automatically resume training if it was interrupted.
        
        Returns:
            Tuple of (resumed, checkpoint_data)
        """
        if self.check_for_interruption():
            self.logger.info("Detected interrupted training, attempting to resume")
            checkpoint_data = self.resume()
            return bool(checkpoint_data), checkpoint_data
        
        return False, {}
    
    def __del__(self):
        """
        Clean up resources.
        """
        try:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
        except (AttributeError, ValueError):
            pass


class InterruptMonitor:
    """
    Monitor for detecting and handling interruptions.
    
    Monitors system resources and network connectivity to detect potential
    interruptions before they occur and save checkpoints proactively.
    
    Attributes:
        checkpoint_manager: Manager for checkpoints
        model: Neural network model
        optimizer: Optimizer for training
        check_interval: How often to check for potential interruptions (in seconds)
        resource_thresholds: Thresholds for system resources
        logger: Logger for monitoring events
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        model: Module,
        optimizer: Optimizer,
        check_interval: float = 10.0,
        resource_thresholds: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize interrupt monitor.
        
        Args:
            checkpoint_manager: Manager for checkpoints
            model: Neural network model
            optimizer: Optimizer for training
            check_interval: How often to check for potential interruptions (in seconds)
            resource_thresholds: Thresholds for system resources
            logger: Logger for monitoring events
        """
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.optimizer = optimizer
        self.check_interval = check_interval
        self.resource_thresholds = resource_thresholds or {
            "memory": 0.95,  # 95% memory usage
            "cpu": 0.95,     # 95% CPU usage
            "disk": 0.95     # 95% disk usage
        }
        self.logger = logger or logging.getLogger(__name__)
        
        self.monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def start_monitoring(self):
        """
        Start monitoring for potential interruptions.
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Interrupt monitoring started")
    
    def stop_monitoring(self):
        """
        Stop monitoring for potential interruptions.
        """
        if not self.monitoring:
            return
        
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.monitoring = False
        self.logger.info("Interrupt monitoring stopped")
    
    def _monitor_loop(self):
        """
        Main monitoring loop.
        """
        if self._has_phynexus:
            self._phynexus.async_train.monitor_interruptions(
                self.checkpoint_manager, self.model, self.optimizer,
                self.check_interval, self.resource_thresholds,
                self.stop_event
            )
            return
        
        while not self.stop_event.is_set():
            try:
                if self._check_resources():
                    self.logger.warning("Potential interruption detected, saving checkpoint proactively")
                    self.checkpoint_manager.save(
                        "main",
                        self.model,
                        self.optimizer,
                        metadata={
                            "proactive_save": True,
                            "reason": "resource_threshold",
                            "time": time.time()
                        }
                    )
            
            except Exception as e:
                self.logger.error(f"Error in interrupt monitor: {str(e)}")
            
            self.stop_event.wait(self.check_interval)
    
    def _check_resources(self) -> bool:
        """
        Check system resources against thresholds.
        
        Returns:
            True if any resource exceeds its threshold, False otherwise
        """
        try:
            import psutil
            
            memory_percent = psutil.virtual_memory().percent / 100.0
            if memory_percent > self.resource_thresholds.get("memory", 0.95):
                self.logger.warning(f"Memory usage ({memory_percent:.2%}) exceeds threshold ({self.resource_thresholds.get('memory', 0.95):.2%})")
                return True
            
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            if cpu_percent > self.resource_thresholds.get("cpu", 0.95):
                self.logger.warning(f"CPU usage ({cpu_percent:.2%}) exceeds threshold ({self.resource_thresholds.get('cpu', 0.95):.2%})")
                return True
            
            disk_percent = psutil.disk_usage('/').percent / 100.0
            if disk_percent > self.resource_thresholds.get("disk", 0.95):
                self.logger.warning(f"Disk usage ({disk_percent:.2%}) exceeds threshold ({self.resource_thresholds.get('disk', 0.95):.2%})")
                return True
            
            return False
        
        except ImportError:
            self.logger.warning("psutil not available, skipping resource check")
            return False
        except Exception as e:
            self.logger.error(f"Error checking resources: {str(e)}")
            return False
