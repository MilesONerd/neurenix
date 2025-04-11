"""
Kubernetes deployment management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Kubernetes deployments for Neurenix models and applications.
"""

import os
import json
import yaml
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

class Deployment:
    """Kubernetes deployment resource."""
    
    def __init__(
        self,
        name: str,
        namespace: str = "default",
        config: Optional["DeploymentConfig"] = None,
    ):
        """
        Initialize a Kubernetes deployment.
        
        Args:
            name: Name of the deployment
            namespace: Kubernetes namespace
            config: Deployment configuration
        """
        self.name = name
        self.namespace = namespace
        self.config = config or DeploymentConfig(name=name, image="placeholder:latest")
        self._status = {}
    
    def apply(self) -> bool:
        """
        Apply the deployment to the Kubernetes cluster.
        
        Returns:
            True if successful, False otherwise
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write(self.config.to_yaml())
            f.flush()
            
            try:
                subprocess.run(
                    ["kubectl", "apply", "-f", f.name],
                    check=True,
                    capture_output=True,
                    text=True
                )
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error applying deployment: {e.stderr}")
                return False
    
    def delete(self) -> bool:
        """
        Delete the deployment from the Kubernetes cluster.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            subprocess.run(
                ["kubectl", "delete", "deployment", self.name, "-n", self.namespace],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error deleting deployment: {e.stderr}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the deployment.
        
        Returns:
            Deployment status
        """
        try:
            result = subprocess.run(
                ["kubectl", "get", "deployment", self.name, "-n", self.namespace, "-o", "json"],
                check=True,
                capture_output=True,
                text=True
            )
            self._status = json.loads(result.stdout)
            return self._status
        except subprocess.CalledProcessError as e:
            print(f"Error getting deployment status: {e.stderr}")
            return {}
    
    def is_ready(self) -> bool:
        """
        Check if the deployment is ready.
        
        Returns:
            True if ready, False otherwise
        """
        status = self.get_status()
        if not status:
            return False
        
        if "status" not in status:
            return False
        
        if "readyReplicas" not in status["status"]:
            return False
        
        return status["status"]["readyReplicas"] == status["status"]["replicas"]
    
    def scale(self, replicas: int) -> bool:
        """
        Scale the deployment.
        
        Args:
            replicas: Number of replicas
            
        Returns:
            True if successful, False otherwise
        """
        try:
            subprocess.run(
                ["kubectl", "scale", "deployment", self.name, "-n", self.namespace, f"--replicas={replicas}"],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error scaling deployment: {e.stderr}")
            return False
    
    def restart(self) -> bool:
        """
        Restart the deployment.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            subprocess.run(
                ["kubectl", "rollout", "restart", "deployment", self.name, "-n", self.namespace],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error restarting deployment: {e.stderr}")
            return False

class DeploymentConfig:
    """Configuration for a Kubernetes deployment."""
    
    def __init__(
        self,
        name: str,
        image: str,
        replicas: int = 1,
        namespace: str = "default",
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        env_from: Optional[List[Dict[str, str]]] = None,
        ports: Optional[List[Dict[str, Any]]] = None,
        volume_mounts: Optional[List[Dict[str, str]]] = None,
        volumes: Optional[List[Dict[str, Any]]] = None,
        resources: Optional[Dict[str, Dict[str, str]]] = None,
        node_selector: Optional[Dict[str, str]] = None,
        tolerations: Optional[List[Dict[str, Any]]] = None,
        affinity: Optional[Dict[str, Any]] = None,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        liveness_probe: Optional[Dict[str, Any]] = None,
        readiness_probe: Optional[Dict[str, Any]] = None,
        startup_probe: Optional[Dict[str, Any]] = None,
        security_context: Optional[Dict[str, Any]] = None,
        service_account: Optional[str] = None,
        image_pull_secrets: Optional[List[Dict[str, str]]] = None,
        strategy: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a deployment configuration.
        
        Args:
            name: Name of the deployment
            image: Docker image to use
            replicas: Number of replicas
            namespace: Kubernetes namespace
            labels: Labels to apply to the deployment
            annotations: Annotations to apply to the deployment
            env: Environment variables
            env_from: Environment variables from ConfigMaps or Secrets
            ports: Container ports
            volume_mounts: Volume mounts
            volumes: Volumes
            resources: Resource requests and limits
            node_selector: Node selector
            tolerations: Tolerations
            affinity: Affinity
            command: Container command
            args: Container arguments
            liveness_probe: Liveness probe
            readiness_probe: Readiness probe
            startup_probe: Startup probe
            security_context: Security context
            service_account: Service account
            image_pull_secrets: Image pull secrets
            strategy: Deployment strategy
        """
        self.name = name
        self.image = image
        self.replicas = replicas
        self.namespace = namespace
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.env = env or {}
        self.env_from = env_from or []
        self.ports = ports or []
        self.volume_mounts = volume_mounts or []
        self.volumes = volumes or []
        self.resources = resources or {}
        self.node_selector = node_selector or {}
        self.tolerations = tolerations or []
        self.affinity = affinity or {}
        self.command = command
        self.args = args
        self.liveness_probe = liveness_probe
        self.readiness_probe = readiness_probe
        self.startup_probe = startup_probe
        self.security_context = security_context
        self.service_account = service_account
        self.image_pull_secrets = image_pull_secrets or []
        self.strategy = strategy or {"type": "RollingUpdate"}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the deployment
        """
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
                "annotations": self.annotations
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {
                    "matchLabels": {"app": self.name}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": self.name, **self.labels},
                        "annotations": self.annotations
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": self.name,
                                "image": self.image
                            }
                        ]
                    }
                },
                "strategy": self.strategy
            }
        }
        
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        
        if self.env:
            container["env"] = [{"name": k, "value": v} for k, v in self.env.items()]
        
        if self.env_from:
            container["envFrom"] = self.env_from
        
        if self.ports:
            container["ports"] = self.ports
        
        if self.volume_mounts:
            container["volumeMounts"] = self.volume_mounts
        
        if self.volumes:
            deployment["spec"]["template"]["spec"]["volumes"] = self.volumes
        
        if self.resources:
            container["resources"] = self.resources
        
        if self.node_selector:
            deployment["spec"]["template"]["spec"]["nodeSelector"] = self.node_selector
        
        if self.tolerations:
            deployment["spec"]["template"]["spec"]["tolerations"] = self.tolerations
        
        if self.affinity:
            deployment["spec"]["template"]["spec"]["affinity"] = self.affinity
        
        if self.command:
            container["command"] = self.command
        
        if self.args:
            container["args"] = self.args
        
        if self.liveness_probe:
            container["livenessProbe"] = self.liveness_probe
        
        if self.readiness_probe:
            container["readinessProbe"] = self.readiness_probe
        
        if self.startup_probe:
            container["startupProbe"] = self.startup_probe
        
        if self.security_context:
            container["securityContext"] = self.security_context
        
        if self.service_account:
            deployment["spec"]["template"]["spec"]["serviceAccountName"] = self.service_account
        
        if self.image_pull_secrets:
            deployment["spec"]["template"]["spec"]["imagePullSecrets"] = self.image_pull_secrets
        
        return deployment
    
    def to_yaml(self) -> str:
        """
        Convert the configuration to YAML.
        
        Returns:
            YAML representation of the deployment
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)
