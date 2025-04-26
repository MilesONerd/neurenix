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
