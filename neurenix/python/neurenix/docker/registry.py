"""
Docker registry management for Neurenix.

This module provides classes and functions for interacting with Docker
registries for Neurenix models and applications.
"""

import json
import subprocess
import base64
from typing import Dict, List, Optional, Any

class RegistryAuth:
    """Docker registry authentication."""
    
    def __init__(self, username: str, password: str, server: str = "https://index.docker.io/v1/"):
        """
        Initialize registry authentication.
        
        Args:
            username: Registry username
            password: Registry password
            server: Registry server URL
        """
        self.username = username
        self.password = password
        self.server = server
    
    def to_auth_config(self) -> Dict[str, Any]:
        """
        Convert to Docker auth config.
        
        Returns:
            Auth config dictionary
        """
        auth = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        return {
            "auths": {
                self.server: {
                    "auth": auth
                }
            }
        }
    
    def login(self) -> None:
        """Log in to the registry."""
        subprocess.run(["docker", "login", "-u", self.username, "-p", self.password, self.server], check=True)
    
    def logout(self) -> None:
        """Log out from the registry."""
        subprocess.run(["docker", "logout", self.server], check=True)


class Registry:
    """Docker registry management for Neurenix."""
    
    def __init__(self, url: str):
        """
        Initialize a registry.
        
        Args:
            url: Registry URL
        """
        self.url = url
        self._check_docker()
    
    def _check_docker(self):
        """Check if Docker is installed and running."""
        try:
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker is not installed or not running")
    
    def push(self, image: str, auth: Optional[RegistryAuth] = None) -> None:
        """
        Push an image to the registry.
        
        Args:
            image: Image name
            auth: Registry authentication
        """
        if auth:
            auth.login()
        
        try:
            subprocess.run(["docker", "push", f"{self.url}/{image}"], check=True)
        finally:
            if auth:
                auth.logout()
    
    def pull(self, image: str, auth: Optional[RegistryAuth] = None) -> None:
        """
        Pull an image from the registry.
        
        Args:
            image: Image name
            auth: Registry authentication
        """
        if auth:
            auth.login()
        
        try:
            subprocess.run(["docker", "pull", f"{self.url}/{image}"], check=True)
        finally:
            if auth:
                auth.logout()
    
    def tag(self, image: str, tag: str) -> None:
        """
        Tag an image for the registry.
        
        Args:
            image: Image name
            tag: Tag name
        """
        subprocess.run(["docker", "tag", image, f"{self.url}/{tag}"], check=True)
    
    def list_images(self, auth: Optional[RegistryAuth] = None) -> List[Dict[str, Any]]:
        """
        List images in the registry.
        
        Args:
            auth: Registry authentication
            
        Returns:
            List of images
        """
        if auth:
            auth.login()
        
        try:
            result = subprocess.run(["docker", "search", self.url], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            images = []
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    images.append({
                        "name": parts[0],
                        "description": " ".join(parts[1:-1]),
                        "official": "[OK]" in parts[-1],
                        "stars": int(parts[-1]) if parts[-1].isdigit() else 0
                    })
            
            return images
        finally:
            if auth:
                auth.logout()
