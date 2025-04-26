"""
Kubernetes Job management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Kubernetes Jobs for Neurenix models and applications.
"""

import os
import json
import yaml
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

class JobConfig:
    """Configuration for a Kubernetes Job."""
    
    def __init__(
        self,
        name: str,
        image: str,
        namespace: str = "default",
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        env_from: Optional[List[Dict[str, str]]] = None,
        volume_mounts: Optional[List[Dict[str, str]]] = None,
        volumes: Optional[List[Dict[str, Any]]] = None,
        resources: Optional[Dict[str, Dict[str, str]]] = None,
        node_selector: Optional[Dict[str, str]] = None,
        tolerations: Optional[List[Dict[str, Any]]] = None,
        affinity: Optional[Dict[str, Any]] = None,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        security_context: Optional[Dict[str, Any]] = None,
        service_account: Optional[str] = None,
        image_pull_secrets: Optional[List[Dict[str, str]]] = None,
        restart_policy: str = "Never",
        backoff_limit: int = 6,
        active_deadline_seconds: Optional[int] = None,
        ttl_seconds_after_finished: Optional[int] = None,
        completions: Optional[int] = None,
        parallelism: Optional[int] = None,
        completion_mode: Optional[str] = None,
    ):
        """
        Initialize a Job configuration.
        
        Args:
            name: Name of the Job
            image: Docker image to use
            namespace: Kubernetes namespace
            labels: Labels to apply to the Job
            annotations: Annotations to apply to the Job
            env: Environment variables
            env_from: Environment variables from ConfigMaps or Secrets
            volume_mounts: Volume mounts
            volumes: Volumes
            resources: Resource requests and limits
            node_selector: Node selector
            tolerations: Tolerations
            affinity: Affinity
            command: Container command
            args: Container arguments
            security_context: Security context
            service_account: Service account
            image_pull_secrets: Image pull secrets
            restart_policy: Restart policy (Never, OnFailure)
            backoff_limit: Number of retries before considering a Job as failed
            active_deadline_seconds: Time limit for running Job
            ttl_seconds_after_finished: Time to keep Job after it finishes
            completions: Number of successful completions required
            parallelism: Number of pods to run in parallel
            completion_mode: Completion mode (Indexed, NonIndexed)
        """
        self.name = name
        self.image = image
        self.namespace = namespace
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.env = env or {}
        self.env_from = env_from or []
        self.volume_mounts = volume_mounts or []
        self.volumes = volumes or []
        self.resources = resources or {}
        self.node_selector = node_selector or {}
        self.tolerations = tolerations or []
        self.affinity = affinity or {}
        self.command = command
        self.args = args
        self.security_context = security_context
        self.service_account = service_account
        self.image_pull_secrets = image_pull_secrets or []
        self.restart_policy = restart_policy
        self.backoff_limit = backoff_limit
        self.active_deadline_seconds = active_deadline_seconds
        self.ttl_seconds_after_finished = ttl_seconds_after_finished
        self.completions = completions
        self.parallelism = parallelism
        self.completion_mode = completion_mode
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the Job
        """
        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
                "annotations": self.annotations
            },
            "spec": {
                "template": {
                    "metadata": {
                        "labels": {"job": self.name, **self.labels},
                        "annotations": self.annotations
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": self.name,
                                "image": self.image
                            }
                        ],
                        "restartPolicy": self.restart_policy
                    }
                },
                "backoffLimit": self.backoff_limit
            }
        }
        
        container = job["spec"]["template"]["spec"]["containers"][0]
        
        if self.env:
            container["env"] = [{"name": k, "value": v} for k, v in self.env.items()]
        
        if self.env_from:
            container["envFrom"] = self.env_from
        
        if self.volume_mounts:
            container["volumeMounts"] = self.volume_mounts
        
        if self.volumes:
            job["spec"]["template"]["spec"]["volumes"] = self.volumes
        
        if self.resources:
            container["resources"] = self.resources
        
        if self.node_selector:
            job["spec"]["template"]["spec"]["nodeSelector"] = self.node_selector
        
        if self.tolerations:
            job["spec"]["template"]["spec"]["tolerations"] = self.tolerations
        
        if self.affinity:
            job["spec"]["template"]["spec"]["affinity"] = self.affinity
        
        if self.command:
            container["command"] = self.command
        
        if self.args:
            container["args"] = self.args
        
        if self.security_context:
            container["securityContext"] = self.security_context
        
        if self.service_account:
            job["spec"]["template"]["spec"]["serviceAccountName"] = self.service_account
        
        if self.image_pull_secrets:
            job["spec"]["template"]["spec"]["imagePullSecrets"] = self.image_pull_secrets
        
        if self.active_deadline_seconds:
            job["spec"]["activeDeadlineSeconds"] = self.active_deadline_seconds
        
        if self.ttl_seconds_after_finished:
            job["spec"]["ttlSecondsAfterFinished"] = self.ttl_seconds_after_finished
        
        if self.completions:
            job["spec"]["completions"] = self.completions
        
        if self.parallelism:
            job["spec"]["parallelism"] = self.parallelism
        
        if self.completion_mode:
            job["spec"]["completionMode"] = self.completion_mode
        
        return job
    
    def to_yaml(self) -> str:
        """
        Convert the configuration to YAML.
        
        Returns:
            YAML representation of the Job
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)


class Job:
    """Kubernetes Job management for Neurenix."""
    
    def __init__(self, name: str, namespace: str = "default"):
        """
        Initialize a Job.
        
        Args:
            name: Name of the Job
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
    
    def create(self, config: JobConfig) -> None:
        """
        Create a Job.
        
        Args:
            config: Job configuration
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write(config.to_yaml())
            f.flush()
            subprocess.run(["kubectl", "apply", "-f", f.name], check=True)
    
    def delete(self) -> None:
        """Delete the Job."""
        subprocess.run(["kubectl", "delete", "job", self.name, "-n", self.namespace], check=True)
    
    def exists(self) -> bool:
        """
        Check if the Job exists.
        
        Returns:
            Whether the Job exists
        """
        try:
            subprocess.run(
                ["kubectl", "get", "job", self.name, "-n", self.namespace],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get(self) -> Dict[str, Any]:
        """
        Get the Job.
        
        Returns:
            Job information
        """
        result = subprocess.run(
            ["kubectl", "get", "job", self.name, "-n", self.namespace, "-o", "json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return json.loads(result.stdout)
    
    def status(self) -> Dict[str, Any]:
        """
        Get the Job status.
        
        Returns:
            Job status
        """
        job = self.get()
        
        if "status" in job:
            return job["status"]
        
        return {}
    
    def is_complete(self) -> bool:
        """
        Check if the Job is complete.
        
        Returns:
            Whether the Job is complete
        """
        status = self.status()
        
        if "succeeded" in status and status["succeeded"] > 0:
            return True
        
        return False
    
    def is_failed(self) -> bool:
        """
        Check if the Job has failed.
        
        Returns:
            Whether the Job has failed
        """
        status = self.status()
        
        if "failed" in status and status["failed"] > 0:
            return True
        
        return False
    
    def logs(self, container: Optional[str] = None, follow: bool = False, tail: Optional[int] = None) -> str:
        """
        Get Job logs.
        
        Args:
            container: Container name
            follow: Whether to follow the logs
            tail: Number of lines to show from the end of the logs
            
        Returns:
            Job logs
        """
        cmd = ["kubectl", "logs", f"job/{self.name}", "-n", self.namespace]
        
        if container:
            cmd.extend(["-c", container])
        
        if follow:
            cmd.append("-f")
        
        if tail:
            cmd.extend(["--tail", str(tail)])
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    
    def create_neurenix_training_job(
        self,
        image: str,
        model_path: str,
        dataset_path: str,
        output_path: str,
        gpu: bool = False,
        memory: Optional[str] = None,
        cpu: Optional[str] = None,
        hyperparameters: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        volume_mounts: Optional[List[Dict[str, str]]] = None,
        volumes: Optional[List[Dict[str, Any]]] = None,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        backoff_limit: int = 3,
        ttl_seconds_after_finished: int = 3600,
    ) -> None:
        """
        Create a Neurenix training Job.
        
        Args:
            image: Docker image to use
            model_path: Path to the model in the container
            dataset_path: Path to the dataset in the container
            output_path: Path to the output directory in the container
            gpu: Whether to use GPU
            memory: Memory request and limit
            cpu: CPU request and limit
            hyperparameters: Hyperparameters for training
            env: Environment variables
            volume_mounts: Volume mounts
            volumes: Volumes
            command: Container command
            args: Container arguments
            backoff_limit: Number of retries before considering a Job as failed
            ttl_seconds_after_finished: Time to keep Job after it finishes
        """
        job_env = {
            "MODEL_PATH": model_path,
            "DATASET_PATH": dataset_path,
            "OUTPUT_PATH": output_path,
            **(env or {})
        }
        
        if hyperparameters:
            for k, v in hyperparameters.items():
                job_env[f"HP_{k.upper()}"] = v
        
        config = JobConfig(
            name=self.name,
            image=image,
            namespace=self.namespace,
            labels={"app": self.name, "component": "neurenix-training"},
            annotations={"neurenix.ai/model-path": model_path, "neurenix.ai/dataset-path": dataset_path},
            env=job_env,
            volume_mounts=volume_mounts,
            volumes=volumes,
            resources={
                "requests": {},
                "limits": {}
            },
            command=command,
            args=args,
            backoff_limit=backoff_limit,
            ttl_seconds_after_finished=ttl_seconds_after_finished
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
    
    def create_neurenix_inference_job(
        self,
        image: str,
        model_path: str,
        input_path: str,
        output_path: str,
        gpu: bool = False,
        memory: Optional[str] = None,
        cpu: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        volume_mounts: Optional[List[Dict[str, str]]] = None,
        volumes: Optional[List[Dict[str, Any]]] = None,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        backoff_limit: int = 3,
        ttl_seconds_after_finished: int = 3600,
    ) -> None:
        """
        Create a Neurenix inference Job.
        
        Args:
            image: Docker image to use
            model_path: Path to the model in the container
            input_path: Path to the input data in the container
            output_path: Path to the output directory in the container
            gpu: Whether to use GPU
            memory: Memory request and limit
            cpu: CPU request and limit
            env: Environment variables
            volume_mounts: Volume mounts
            volumes: Volumes
            command: Container command
            args: Container arguments
            backoff_limit: Number of retries before considering a Job as failed
            ttl_seconds_after_finished: Time to keep Job after it finishes
        """
        job_env = {
            "MODEL_PATH": model_path,
            "INPUT_PATH": input_path,
            "OUTPUT_PATH": output_path,
            "MODE": "inference",
            **(env or {})
        }
        
        config = JobConfig(
            name=self.name,
            image=image,
            namespace=self.namespace,
            labels={"app": self.name, "component": "neurenix-inference"},
            annotations={"neurenix.ai/model-path": model_path, "neurenix.ai/input-path": input_path},
            env=job_env,
            volume_mounts=volume_mounts,
            volumes=volumes,
            resources={
                "requests": {},
                "limits": {}
            },
            command=command,
            args=args,
            backoff_limit=backoff_limit,
            ttl_seconds_after_finished=ttl_seconds_after_finished
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
