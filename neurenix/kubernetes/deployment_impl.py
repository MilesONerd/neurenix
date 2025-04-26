"""
Kubernetes deployment implementation for Neurenix.

This module provides the Deployment class for creating, managing, and
interacting with Kubernetes deployments for Neurenix models and applications.
"""

import os
import json
import yaml
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

class Deployment:
    """Kubernetes deployment management for Neurenix."""
    
    def __init__(self, name: str, namespace: str = "default"):
        """
        Initialize a deployment.
        
        Args:
            name: Name of the deployment
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
    
    def create(self, config):
        """
        Create a deployment.
        
        Args:
            config: Deployment configuration
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write(config.to_yaml())
            f.flush()
            subprocess.run(["kubectl", "apply", "-f", f.name], check=True)
    
    def delete(self, wait: bool = False) -> None:
        """
        Delete the deployment.
        
        Args:
            wait: Whether to wait for deletion to complete
        """
        cmd = ["kubectl", "delete", "deployment", self.name, "-n", self.namespace]
        
        if wait:
            cmd.append("--wait")
        
        subprocess.run(cmd, check=True)
    
    def exists(self) -> bool:
        """
        Check if the deployment exists.
        
        Returns:
            Whether the deployment exists
        """
        try:
            subprocess.run(
                ["kubectl", "get", "deployment", self.name, "-n", self.namespace],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get(self) -> Dict[str, Any]:
        """
        Get the deployment.
        
        Returns:
            Deployment information
        """
        result = subprocess.run(
            ["kubectl", "get", "deployment", self.name, "-n", self.namespace, "-o", "json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return json.loads(result.stdout)
    
    def scale(self, replicas: int) -> None:
        """
        Scale the deployment.
        
        Args:
            replicas: Number of replicas
        """
        subprocess.run(
            ["kubectl", "scale", "deployment", self.name, "-n", self.namespace, f"--replicas={replicas}"],
            check=True
        )
    
    def restart(self) -> None:
        """Restart the deployment."""
        subprocess.run(
            ["kubectl", "rollout", "restart", "deployment", self.name, "-n", self.namespace],
            check=True
        )
    
    def status(self) -> Dict[str, Any]:
        """
        Get the deployment status.
        
        Returns:
            Deployment status
        """
        result = subprocess.run(
            ["kubectl", "rollout", "status", "deployment", self.name, "-n", self.namespace, "-o", "json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return json.loads(result.stdout)
    
    def logs(self, container: Optional[str] = None, follow: bool = False, tail: Optional[int] = None) -> str:
        """
        Get deployment logs.
        
        Args:
            container: Container name
            follow: Whether to follow the logs
            tail: Number of lines to show from the end of the logs
            
        Returns:
            Deployment logs
        """
        cmd = ["kubectl", "logs", f"deployment/{self.name}", "-n", self.namespace]
        
        if container:
            cmd.extend(["-c", container])
        
        if follow:
            cmd.append("-f")
        
        if tail:
            cmd.extend(["--tail", str(tail)])
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    
    def exec(self, command: Union[str, List[str]], container: Optional[str] = None) -> str:
        """
        Execute a command in the deployment.
        
        Args:
            command: Command to execute
            container: Container name
            
        Returns:
            Command output
        """
        cmd = ["kubectl", "exec", f"deployment/{self.name}", "-n", self.namespace]
        
        if container:
            cmd.extend(["-c", container])
        
        if isinstance(command, list):
            cmd.extend(["--", *command])
        else:
            cmd.extend(["--", *command.split()])
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    
    def port_forward(self, local_port: int, remote_port: int) -> subprocess.Popen:
        """
        Forward a port to the deployment.
        
        Args:
            local_port: Local port
            remote_port: Remote port
            
        Returns:
            Port forwarding process
        """
        return subprocess.Popen(
            ["kubectl", "port-forward", f"deployment/{self.name}", "-n", self.namespace, f"{local_port}:{remote_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    def update_image(self, image: str) -> None:
        """
        Update the deployment image.
        
        Args:
            image: New image
        """
        subprocess.run(
            ["kubectl", "set", "image", f"deployment/{self.name}", f"{self.name}={image}", "-n", self.namespace],
            check=True
        )
    
    def create_neurenix_deployment(
        self,
        image: str,
        model_path: str,
        replicas: int = 1,
        gpu: bool = False,
        memory: Optional[str] = None,
        cpu: Optional[str] = None,
        port: int = 8000,
        env: Optional[Dict[str, str]] = None,
        volume_mounts: Optional[List[Dict[str, str]]] = None,
        volumes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Create a Neurenix deployment.
        
        Args:
            image: Docker image to use
            model_path: Path to the model in the container
            replicas: Number of replicas
            gpu: Whether to use GPU
            memory: Memory request and limit
            cpu: CPU request and limit
            port: Container port
            env: Environment variables
            volume_mounts: Volume mounts
            volumes: Volumes
        """
        from neurenix.kubernetes.deployment import DeploymentConfig
        
        config = DeploymentConfig(
            name=self.name,
            image=image,
            replicas=replicas,
            namespace=self.namespace,
            labels={"app": self.name, "component": "neurenix"},
            annotations={"neurenix.ai/model-path": model_path},
            env={
                "MODEL_PATH": model_path,
                "PORT": str(port),
                **(env or {})
            },
            ports=[{"containerPort": port, "protocol": "TCP"}],
            volume_mounts=volume_mounts,
            volumes=volumes,
            resources={
                "requests": {},
                "limits": {}
            }
        )
        
        if memory:
            config.resources["requests"]["memory"] = memory
            config.resources["limits"]["memory"] = memory
        
        if cpu:
            config.resources["requests"]["cpu"] = cpu
            config.resources["limits"]["cpu"] = cpu
        
        if gpu:
            config.resources["limits"]["nvidia.com/gpu"] = "1"
            config.node_selector = {"accelerator": "nvidia"}
        
        self.create(config)
