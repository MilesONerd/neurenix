"""
Kubernetes Secret management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Kubernetes Secrets for Neurenix models and applications.
"""

import os
import json
import yaml
import base64
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

class Secret:
    """Kubernetes Secret management for Neurenix."""
    
    def __init__(self, name: str, namespace: str = "default"):
        """
        Initialize a Secret.
        
        Args:
            name: Name of the Secret
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
               labels: Optional[Dict[str, str]] = None, annotations: Optional[Dict[str, str]] = None,
               type: str = "Opaque") -> None:
        """
        Create a Secret.
        
        Args:
            data: Secret data (values will be base64 encoded)
            from_file: Files to include in the Secret
            from_env_file: Environment file to include in the Secret
            from_literal: Literal values to include in the Secret
            labels: Labels to apply to the Secret
            annotations: Annotations to apply to the Secret
            type: Secret type (Opaque, kubernetes.io/tls, etc.)
        """
        if data:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
                encoded_data = {}
                for k, v in data.items():
                    encoded_data[k] = base64.b64encode(v.encode()).decode()
                
                secret = {
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "metadata": {
                        "name": self.name,
                        "namespace": self.namespace
                    },
                    "type": type,
                    "data": encoded_data
                }
                
                if labels:
                    secret["metadata"]["labels"] = labels
                
                if annotations:
                    secret["metadata"]["annotations"] = annotations
                
                f.write(yaml.dump(secret, default_flow_style=False))
                f.flush()
                subprocess.run(["kubectl", "apply", "-f", f.name], check=True)
        else:
            cmd = ["kubectl", "create", "secret", "generic", self.name, "-n", self.namespace]
            
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
            
            if type != "Opaque":
                cmd.extend(["--type", type])
            
            subprocess.run(cmd, check=True)
    
    def create_tls(self, cert_file: str, key_file: str, labels: Optional[Dict[str, str]] = None, 
                  annotations: Optional[Dict[str, str]] = None) -> None:
        """
        Create a TLS Secret.
        
        Args:
            cert_file: Path to the certificate file
            key_file: Path to the key file
            labels: Labels to apply to the Secret
            annotations: Annotations to apply to the Secret
        """
        cmd = ["kubectl", "create", "secret", "tls", self.name, "-n", self.namespace,
               "--cert", cert_file, "--key", key_file]
        
        if labels:
            for k, v in labels.items():
                cmd.extend(["--labels", f"{k}={v}"])
        
        subprocess.run(cmd, check=True)
    
    def create_docker_registry(self, server: str, username: str, password: str, email: Optional[str] = None,
                              labels: Optional[Dict[str, str]] = None, annotations: Optional[Dict[str, str]] = None) -> None:
        """
        Create a Docker registry Secret.
        
        Args:
            server: Docker registry server
            username: Docker registry username
            password: Docker registry password
            email: Docker registry email
            labels: Labels to apply to the Secret
            annotations: Annotations to apply to the Secret
        """
        cmd = ["kubectl", "create", "secret", "docker-registry", self.name, "-n", self.namespace,
               "--docker-server", server, "--docker-username", username, "--docker-password", password]
        
        if email:
            cmd.extend(["--docker-email", email])
        
        if labels:
            for k, v in labels.items():
                cmd.extend(["--labels", f"{k}={v}"])
        
        subprocess.run(cmd, check=True)
    
    def delete(self) -> None:
        """Delete the Secret."""
        subprocess.run(["kubectl", "delete", "secret", self.name, "-n", self.namespace], check=True)
    
    def exists(self) -> bool:
        """
        Check if the Secret exists.
        
        Returns:
            Whether the Secret exists
        """
        try:
            subprocess.run(
                ["kubectl", "get", "secret", self.name, "-n", self.namespace],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get(self) -> Dict[str, Any]:
        """
        Get the Secret.
        
        Returns:
            Secret information
        """
        result = subprocess.run(
            ["kubectl", "get", "secret", self.name, "-n", self.namespace, "-o", "json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return json.loads(result.stdout)
    
    def get_data(self, decode: bool = True) -> Dict[str, str]:
        """
        Get the Secret data.
        
        Args:
            decode: Whether to base64 decode the data values
            
        Returns:
            Secret data
        """
        secret = self.get()
        
        if "data" in secret:
            data = secret["data"]
            
            if decode:
                decoded_data = {}
                for k, v in data.items():
                    decoded_data[k] = base64.b64decode(v).decode()
                return decoded_data
            
            return data
        
        return {}
    
    def update(self, data: Dict[str, str]) -> None:
        """
        Update the Secret data.
        
        Args:
            data: New Secret data (values will be base64 encoded)
        """
        secret = self.get()
        
        encoded_data = {}
        for k, v in data.items():
            encoded_data[k] = base64.b64encode(v.encode()).decode()
        
        if "data" in secret:
            secret["data"].update(encoded_data)
        else:
            secret["data"] = encoded_data
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write(yaml.dump(secret, default_flow_style=False))
            f.flush()
            subprocess.run(["kubectl", "apply", "-f", f.name], check=True)
    
    def create_neurenix_credentials(self, api_key: str, api_secret: Optional[str] = None, 
                                   model_registry_token: Optional[str] = None, 
                                   labels: Optional[Dict[str, str]] = None) -> None:
        """
        Create a Neurenix credentials Secret.
        
        Args:
            api_key: Neurenix API key
            api_secret: Neurenix API secret
            model_registry_token: Neurenix model registry token
            labels: Labels to apply to the Secret
        """
        data = {
            "api-key": api_key
        }
        
        if api_secret:
            data["api-secret"] = api_secret
        
        if model_registry_token:
            data["model-registry-token"] = model_registry_token
        
        self.create(
            data=data,
            labels={"app": self.name, "component": "neurenix", **(labels or {})},
            annotations={"neurenix.ai/credentials": "true"}
        )
