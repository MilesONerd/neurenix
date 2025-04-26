"""
Docker network management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Docker networks for Neurenix models and applications.
"""

import json
import subprocess
from typing import Dict, List, Optional, Any

class Network:
    """Docker network management for Neurenix."""
    
    def __init__(self, name: str):
        """
        Initialize a network.
        
        Args:
            name: Name of the network
        """
        self.name = name
        self._check_docker()
    
    def _check_docker(self):
        """Check if Docker is installed and running."""
        try:
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker is not installed or not running")
    
    def create(self, driver: Optional[str] = None, subnet: Optional[str] = None, gateway: Optional[str] = None, internal: bool = False, labels: Optional[Dict[str, str]] = None, options: Optional[Dict[str, str]] = None) -> None:
        """
        Create the network.
        
        Args:
            driver: Network driver to use
            subnet: Subnet in CIDR format
            gateway: Gateway for the subnet
            internal: Whether the network is internal
            labels: Labels to apply to the network
            options: Driver-specific options
        """
        cmd = ["docker", "network", "create"]
        
        if driver:
            cmd.extend(["--driver", driver])
        
        if subnet:
            cmd.extend(["--subnet", subnet])
        
        if gateway:
            cmd.extend(["--gateway", gateway])
        
        if internal:
            cmd.append("--internal")
        
        if labels:
            for k, v in labels.items():
                cmd.extend(["--label", f"{k}={v}"])
        
        if options:
            for k, v in options.items():
                cmd.extend(["--opt", f"{k}={v}"])
        
        cmd.append(self.name)
        
        subprocess.run(cmd, check=True)
    
    def remove(self) -> None:
        """Remove the network."""
        subprocess.run(["docker", "network", "rm", self.name], check=True)
    
    def exists(self) -> bool:
        """
        Check if the network exists.
        
        Returns:
            Whether the network exists
        """
        try:
            subprocess.run(["docker", "network", "inspect", self.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def inspect(self) -> Dict[str, Any]:
        """
        Inspect the network.
        
        Returns:
            Network information
        """
        result = subprocess.run(["docker", "network", "inspect", self.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return json.loads(result.stdout)[0]
    
    def connect(self, container: str, ip: Optional[str] = None, alias: Optional[str] = None) -> None:
        """
        Connect a container to the network.
        
        Args:
            container: Container ID or name
            ip: IP address to assign to the container
            alias: Network alias for the container
        """
        cmd = ["docker", "network", "connect"]
        
        if ip:
            cmd.extend(["--ip", ip])
        
        if alias:
            cmd.extend(["--alias", alias])
        
        cmd.extend([self.name, container])
        
        subprocess.run(cmd, check=True)
    
    def disconnect(self, container: str, force: bool = False) -> None:
        """
        Disconnect a container from the network.
        
        Args:
            container: Container ID or name
            force: Whether to force disconnection
        """
        cmd = ["docker", "network", "disconnect"]
        
        if force:
            cmd.append("-f")
        
        cmd.extend([self.name, container])
        
        subprocess.run(cmd, check=True)
    
    @classmethod
    def list(cls, filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        List networks.
        
        Args:
            filters: Filters to apply
            
        Returns:
            List of networks
        """
        cmd = ["docker", "network", "ls", "--format", "{{json .}}"]
        
        if filters:
            for k, v in filters.items():
                cmd.extend(["--filter", f"{k}={v}"])
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    
    @classmethod
    def prune(cls, filters: Optional[Dict[str, str]] = None) -> None:
        """
        Remove unused networks.
        
        Args:
            filters: Filters to apply
        """
        cmd = ["docker", "network", "prune", "-f"]
        
        if filters:
            for k, v in filters.items():
                cmd.extend(["--filter", f"{k}={v}"])
        
        subprocess.run(cmd, check=True)
