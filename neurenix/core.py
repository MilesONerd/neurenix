"""
Core functionality for the Neurenix framework.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neurenix")

# Global configuration
_config: Dict[str, Any] = {
    "device": "cpu",
    "debug": False,
    "log_level": "info",
    "tpu_visible_devices": None,  # Control which TPU devices are visible
}

def init(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Initialize the Neurenix framework with the given configuration.
    
    Args:
        config: Configuration dictionary with options for the framework.
    """
    global _config
    
    if config is not None:
        _config.update(config)
    
    # Set up logging based on configuration
    log_level = _config.get("log_level", "info").lower()
    if log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif log_level == "info":
        logger.setLevel(logging.INFO)
    elif log_level == "warning":
        logger.setLevel(logging.WARNING)
    elif log_level == "error":
        logger.setLevel(logging.ERROR)
    
    logger.info(f"Neurenix v{version()} initialized")
    logger.debug(f"Configuration: {_config}")
    
    try:
        from neurenix.binding import init
        init()
        logger.info("Phynexus engine initialized successfully")
    except (ImportError, AttributeError):
        logger.warning("Phynexus engine not available, using fallback implementations")

def version() -> str:
    """
    Get the version of the Neurenix framework.
    
    Returns:
        The version string.
    """
    return "2.0.1"

def get_config() -> Dict[str, Any]:
    """
    Get the current configuration of the Neurenix framework.
    
    Returns:
        The configuration dictionary.
    """
    return _config.copy()

def set_config(key: str, value: Any) -> None:
    """
    Set a configuration option for the Neurenix framework.
    
    Args:
        key: The configuration key.
        value: The configuration value.
    """
    _config[key] = value
    logger.debug(f"Configuration updated: {key}={value}")
