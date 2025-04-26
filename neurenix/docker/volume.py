"""
Docker volume management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Docker volumes for Neurenix models and applications.
"""

import json
import subprocess
from typing import Dict, List, Optional, Any

class Volume:
    """Docker volume management for Neurenix."""
    
    def __init__(self, name: str):
        """
        Initialize a volume.
        
        Args:
            name: Name of the volume
        """
        self.name = name
        self._check_docker()
    
    def _check_docker(self):
        """Check if Docker is installed and running."""
        try:
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker is not installed or not running")
    
    def create(self, driver: Optional[str] = None, labels: Optional[Dict[str, str]] = None, options: Optional[Dict[str, str]] = None) -> None:
        """
        Create the volume.
        
        Args:
            driver: Volume driver to use
            labels: Labels to apply to the volume
            options: Driver-specific options
        """
        cmd = ["docker", "volume", "create"]
        
        if driver:
            cmd.extend(["--driver", driver])
        
        if labels:
            for k, v in labels.items():
                cmd.extend(["--label", f"{k}={v}"])
        
        if options:
            for k, v in options.items():
                cmd.extend(["--opt", f"{k}={v}"])
        
        cmd.append(self.name)
        
        subprocess.run(cmd, check=True)
    
    def remove(self, force: bool = False) -> None:
        """
        Remove the volume.
        
        Args:
            force: Whether to force removal
        """
        cmd = ["docker", "volume", "rm"]
        
        if force:
            cmd.append("-f")
        
        cmd.append(self.name)
        
        subprocess.run(cmd, check=True)
    
    def exists(self) -> bool:
        """
        Check if the volume exists.
        
        Returns:
            Whether the volume exists
        """
        try:
            subprocess.run(["docker", "volume", "inspect", self.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def inspect(self) -> Dict[str, Any]:
        """
        Inspect the volume.
        
        Returns:
            Volume information
        """
        result = subprocess.run(["docker", "volume", "inspect", self.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return json.loads(result.stdout)[0]
    
    def prune(cls, filters: Optional[Dict[str, str]] = None) -> None:
        """
        Remove unused volumes.
        
        Args:
            filters: Filters to apply
        """
        cmd = ["docker", "volume", "prune", "-f"]
        
        if filters:
            for k, v in filters.items():
                cmd.extend(["--filter", f"{k}={v}"])
        
        subprocess.run(cmd, check=True)
    
    @classmethod
    def list(cls, filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        List volumes.
        
        Args:
            filters: Filters to apply
            
        Returns:
            List of volumes
        """
        cmd = ["docker", "volume", "ls", "--format", "{{json .}}"]
        
        if filters:
            for k, v in filters.items():
                cmd.extend(["--filter", f"{k}={v}"])
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
