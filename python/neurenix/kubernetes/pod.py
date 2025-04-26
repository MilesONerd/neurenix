"""
Kubernetes pod management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Kubernetes pods for Neurenix models and applications.
"""

import os
import json
import yaml
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

class PodConfig:
    """Configuration for a Kubernetes pod."""
    
    def __init__(
        self,
        name: str,
        image: str,
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
        restart_policy: Optional[str] = None,
        host_network: bool = False,
        host_pid: bool = False,
        host_ipc: bool = False,
        dns_policy: Optional[str] = None,
        dns_config: Optional[Dict[str, Any]] = None,
        termination_grace_period_seconds: Optional[int] = None,
        active_deadline_seconds: Optional[int] = None,
        priority: Optional[int] = None,
        priority_class_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        node_name: Optional[str] = None,
    ):
        """
        Initialize a pod configuration.
        
        Args:
            name: Name of the pod
            image: Docker image to use
            namespace: Kubernetes namespace
            labels: Labels to apply to the pod
            annotations: Annotations to apply to the pod
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
            restart_policy: Restart policy (Always, OnFailure, Never)
            host_network: Whether to use the host's network namespace
            host_pid: Whether to use the host's PID namespace
            host_ipc: Whether to use the host's IPC namespace
            dns_policy: DNS policy
            dns_config: DNS configuration
            termination_grace_period_seconds: Termination grace period in seconds
            active_deadline_seconds: Active deadline in seconds
            priority: Priority
            priority_class_name: Priority class name
            scheduler_name: Scheduler name
            node_name: Node name
        """
        self.name = name
        self.image = image
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
        self.restart_policy = restart_policy
        self.host_network = host_network
        self.host_pid = host_pid
        self.host_ipc = host_ipc
        self.dns_policy = dns_policy
        self.dns_config = dns_config
        self.termination_grace_period_seconds = termination_grace_period_seconds
        self.active_deadline_seconds = active_deadline_seconds
        self.priority = priority
        self.priority_class_name = priority_class_name
        self.scheduler_name = scheduler_name
        self.node_name = node_name
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the pod
        """
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
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
        }
        
        container = pod["spec"]["containers"][0]
        
        if self.env:
            container["env"] = [{"name": k, "value": v} for k, v in self.env.items()]
        
        if self.env_from:
            container["envFrom"] = self.env_from
        
        if self.ports:
            container["ports"] = self.ports
        
        if self.volume_mounts:
            container["volumeMounts"] = self.volume_mounts
        
        if self.volumes:
            pod["spec"]["volumes"] = self.volumes
        
        if self.resources:
            container["resources"] = self.resources
        
        if self.node_selector:
            pod["spec"]["nodeSelector"] = self.node_selector
        
        if self.tolerations:
            pod["spec"]["tolerations"] = self.tolerations
        
        if self.affinity:
            pod["spec"]["affinity"] = self.affinity
        
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
            pod["spec"]["serviceAccountName"] = self.service_account
        
        if self.image_pull_secrets:
            pod["spec"]["imagePullSecrets"] = self.image_pull_secrets
        
        if self.restart_policy:
            pod["spec"]["restartPolicy"] = self.restart_policy
        
        if self.host_network:
            pod["spec"]["hostNetwork"] = self.host_network
        
        if self.host_pid:
            pod["spec"]["hostPID"] = self.host_pid
        
        if self.host_ipc:
            pod["spec"]["hostIPC"] = self.host_ipc
        
        if self.dns_policy:
            pod["spec"]["dnsPolicy"] = self.dns_policy
        
        if self.dns_config:
            pod["spec"]["dnsConfig"] = self.dns_config
        
        if self.termination_grace_period_seconds:
            pod["spec"]["terminationGracePeriodSeconds"] = self.termination_grace_period_seconds
        
        if self.active_deadline_seconds:
            pod["spec"]["activeDeadlineSeconds"] = self.active_deadline_seconds
        
        if self.priority:
            pod["spec"]["priority"] = self.priority
        
        if self.priority_class_name:
            pod["spec"]["priorityClassName"] = self.priority_class_name
        
        if self.scheduler_name:
            pod["spec"]["schedulerName"] = self.scheduler_name
        
        if self.node_name:
            pod["spec"]["nodeName"] = self.node_name
        
        return pod
    
    def to_yaml(self) -> str:
        """
        Convert the configuration to YAML.
        
        Returns:
            YAML representation of the pod
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)


class Pod:
    """Kubernetes pod management for Neurenix."""
    
    def __init__(self, name: str, namespace: str = "default"):
        """
        Initialize a pod.
        
        Args:
            name: Name of the pod
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
    
    def create(self, config: PodConfig) -> None:
        """
        Create a pod.
        
        Args:
            config: Pod configuration
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write(config.to_yaml())
            f.flush()
            subprocess.run(["kubectl", "apply", "-f", f.name], check=True)
    
    def delete(self, grace_period: Optional[int] = None, force: bool = False) -> None:
        """
        Delete the pod.
        
        Args:
            grace_period: Grace period in seconds
            force: Whether to force deletion
        """
        cmd = ["kubectl", "delete", "pod", self.name, "-n", self.namespace]
        
        if grace_period is not None:
            cmd.extend(["--grace-period", str(grace_period)])
        
        if force:
            cmd.append("--force")
        
        subprocess.run(cmd, check=True)
    
    def exists(self) -> bool:
        """
        Check if the pod exists.
        
        Returns:
            Whether the pod exists
        """
        try:
            subprocess.run(
                ["kubectl", "get", "pod", self.name, "-n", self.namespace],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get(self) -> Dict[str, Any]:
        """
        Get the pod.
        
        Returns:
            Pod information
        """
        result = subprocess.run(
            ["kubectl", "get", "pod", self.name, "-n", self.namespace, "-o", "json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return json.loads(result.stdout)
    
    def status(self) -> str:
        """
        Get the pod status.
        
        Returns:
            Pod status
        """
        pod = self.get()
        
        if "status" in pod and "phase" in pod["status"]:
            return pod["status"]["phase"]
        
        return "Unknown"
    
    def logs(self, container: Optional[str] = None, follow: bool = False, tail: Optional[int] = None) -> str:
        """
        Get pod logs.
        
        Args:
            container: Container name
            follow: Whether to follow the logs
            tail: Number of lines to show from the end of the logs
            
        Returns:
            Pod logs
        """
        cmd = ["kubectl", "logs", self.name, "-n", self.namespace]
        
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
        Execute a command in the pod.
        
        Args:
            command: Command to execute
            container: Container name
            
        Returns:
            Command output
        """
        cmd = ["kubectl", "exec", self.name, "-n", self.namespace]
        
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
        Forward a port to the pod.
        
        Args:
            local_port: Local port
            remote_port: Remote port
            
        Returns:
            Port forwarding process
        """
        return subprocess.Popen(
            ["kubectl", "port-forward", self.name, "-n", self.namespace, f"{local_port}:{remote_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    def copy_to(self, local_path: str, remote_path: str, container: Optional[str] = None) -> None:
        """
        Copy a file to the pod.
        
        Args:
            local_path: Local file path
            remote_path: Remote file path
            container: Container name
        """
        cmd = ["kubectl", "cp", local_path, f"{self.namespace}/{self.name}:{remote_path}"]
        
        if container:
            cmd.extend(["-c", container])
        
        subprocess.run(cmd, check=True)
    
    def copy_from(self, remote_path: str, local_path: str, container: Optional[str] = None) -> None:
        """
        Copy a file from the pod.
        
        Args:
            remote_path: Remote file path
            local_path: Local file path
            container: Container name
        """
        cmd = ["kubectl", "cp", f"{self.namespace}/{self.name}:{remote_path}", local_path]
        
        if container:
            cmd.extend(["-c", container])
        
        subprocess.run(cmd, check=True)
    
    def create_neurenix_pod(
        self,
        image: str,
        model_path: str,
        gpu: bool = False,
        memory: Optional[str] = None,
        cpu: Optional[str] = None,
        port: int = 8000,
        env: Optional[Dict[str, str]] = None,
        volume_mounts: Optional[List[Dict[str, str]]] = None,
        volumes: Optional[List[Dict[str, Any]]] = None,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
    ) -> None:
        """
        Create a Neurenix pod.
        
        Args:
            image: Docker image to use
            model_path: Path to the model in the container
            gpu: Whether to use GPU
            memory: Memory request and limit
            cpu: CPU request and limit
            port: Container port
            env: Environment variables
            volume_mounts: Volume mounts
            volumes: Volumes
            command: Container command
            args: Container arguments
        """
        config = PodConfig(
            name=self.name,
            image=image,
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
            },
            command=command,
            args=args
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
