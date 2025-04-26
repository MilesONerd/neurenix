"""
Kubernetes ConfigMap management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Kubernetes ConfigMaps for Neurenix models and applications.
"""

import os
import json
import yaml
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

class ConfigMap:
    """Kubernetes ConfigMap management for Neurenix."""
    
    def __init__(self, name: str, namespace: str = "default"):
        """
        Initialize a ConfigMap.
        
        Args:
            name: Name of the ConfigMap
            namespace: Kubernetes namespace
        """
        self.name = name
        self.namespace = namespace
        self._check_kubectl()
    
    def _check_kubectl(self):
        """Check if kubectl is installed and configured."""
        try:
            subprocess.run(["kubectl", "version", "--client"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("kubectl is not installed or not configured")
    
    def create(self, data: Optional[Dict[str, str]] = None, from_file: Optional[List[str]] = None, 
               from_env_file: Optional[str] = None, from_literal: Optional[Dict[str, str]] = None, 
               labels: Optional[Dict[str, str]] = None, annotations: Optional[Dict[str, str]] = None) -> None:
        """
        Create a ConfigMap.
        
        Args:
            data: ConfigMap data
            from_file: Files to include in the ConfigMap
            from_env_file: Environment file to include in the ConfigMap
            from_literal: Literal values to include in the ConfigMap
            labels: Labels to apply to the ConfigMap
            annotations: Annotations to apply to the ConfigMap
        """
        if data:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
                config_map = {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": self.name,
                        "namespace": self.namespace
                    },
                    "data": data
                }
                
                if labels:
                    config_map["metadata"]["labels"] = labels
                
                if annotations:
                    config_map["metadata"]["annotations"] = annotations
                
                f.write(yaml.dump(config_map, default_flow_style=False))
                f.flush()
                subprocess.run(["kubectl", "apply", "-f", f.name], check=True)
        else:
            cmd = ["kubectl", "create", "configmap", self.name, "-n", self.namespace]
            
            if from_file:
                for file_path in from_file:
                    cmd.extend(["--from-file", file_path])
            
            if from_env_file:
                cmd.extend(["--from-env-file", from_env_file])
            
            if from_literal:
                for k, v in from_literal.items():
                    cmd.extend(["--from-literal", f"{k}={v}"])
            
            if labels:
                for k, v in labels.items():
                    cmd.extend(["--labels", f"{k}={v}"])
            
            subprocess.run(cmd, check=True)
    
    def delete(self) -> None:
        """Delete the ConfigMap."""
        subprocess.run(["kubectl", "delete", "configmap", self.name, "-n", self.namespace], check=True)
    
    def exists(self) -> bool:
        """
        Check if the ConfigMap exists.
        
        Returns:
            Whether the ConfigMap exists
        """
        try:
            subprocess.run(
                ["kubectl", "get", "configmap", self.name, "-n", self.namespace],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get(self) -> Dict[str, Any]:
        """
        Get the ConfigMap.
        
        Returns:
            ConfigMap information
        """
        result = subprocess.run(
            ["kubectl", "get", "configmap", self.name, "-n", self.namespace, "-o", "json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return json.loads(result.stdout)
    
    def get_data(self) -> Dict[str, str]:
        """
        Get the ConfigMap data.
        
        Returns:
            ConfigMap data
        """
        config_map = self.get()
        
        if "data" in config_map:
            return config_map["data"]
        
        return {}
    
    def update(self, data: Dict[str, str]) -> None:
        """
        Update the ConfigMap data.
        
        Args:
            data: New ConfigMap data
        """
        current_data = self.get_data()
        current_data.update(data)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            config_map = self.get()
            config_map["data"] = current_data
            
            f.write(yaml.dump(config_map, default_flow_style=False))
            f.flush()
            subprocess.run(["kubectl", "apply", "-f", f.name], check=True)
