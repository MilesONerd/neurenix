"""
Docker container management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Docker containers for Neurenix models and applications.
"""

import os
import json
import subprocess
from typing import Dict, List, Optional, Union, Any

class ContainerConfig:
    """Configuration for a Docker container."""
    
    def __init__(
        self,
        image: str,
        command: Optional[Union[str, List[str]]] = None,
        entrypoint: Optional[Union[str, List[str]]] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        ports: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
        user: Optional[str] = None,
        network: Optional[str] = None,
        hostname: Optional[str] = None,
        runtime: Optional[str] = None,
        gpu: bool = False,
        memory: Optional[str] = None,
        cpus: Optional[float] = None,
    ):
        """
        Initialize a container configuration.
        
        Args:
            image: Docker image name
            command: Command to run in the container
            entrypoint: Entrypoint for the container
            environment: Environment variables
            volumes: Volume mappings
            ports: Port mappings
            working_dir: Working directory in the container
            user: User to run the container as
            network: Network to connect the container to
            hostname: Container hostname
            runtime: Container runtime (e.g., 'nvidia')
            gpu: Whether to use GPU acceleration
            memory: Memory limit
            cpus: CPU limit
        """
        self.image = image
        self.command = command
        self.entrypoint = entrypoint
        self.environment = environment or {}
        self.volumes = volumes or {}
        self.ports = ports or {}
        self.working_dir = working_dir
        self.user = user
        self.network = network
        self.hostname = hostname
        self.runtime = runtime
        self.gpu = gpu
        self.memory = memory
        self.cpus = cpus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        config = {
            "Image": self.image,
        }
        
        if self.command:
            if isinstance(self.command, list):
                config["Cmd"] = self.command
            else:
                config["Cmd"] = self.command.split()
        
        if self.entrypoint:
            if isinstance(self.entrypoint, list):
                config["Entrypoint"] = self.entrypoint
            else:
                config["Entrypoint"] = self.entrypoint.split()
        
        if self.environment:
            config["Env"] = [f"{k}={v}" for k, v in self.environment.items()]
        
        if self.volumes:
            config["Volumes"] = {v: {} for v in self.volumes.keys()}
            config["HostConfig"] = {"Binds": [f"{h}:{c}:{m}" for c, h_m in self.volumes.items() for h, m in h_m.items()]}
        
        if self.ports:
            exposed_ports = {}
            port_bindings = {}
            
            for container_port, host_port in self.ports.items():
                exposed_ports[container_port] = {}
                port_bindings[container_port] = [{"HostPort": host_port}]
            
            config["ExposedPorts"] = exposed_ports
            if "HostConfig" not in config:
                config["HostConfig"] = {}
            config["HostConfig"]["PortBindings"] = port_bindings
        
        if self.working_dir:
            config["WorkingDir"] = self.working_dir
        
        if self.user:
            config["User"] = self.user
        
        if self.network:
            if "HostConfig" not in config:
                config["HostConfig"] = {}
            config["HostConfig"]["NetworkMode"] = self.network
        
        if self.hostname:
            config["Hostname"] = self.hostname
        
        if self.runtime:
            if "HostConfig" not in config:
                config["HostConfig"] = {}
            config["HostConfig"]["Runtime"] = self.runtime
        
        if self.gpu:
            if "HostConfig" not in config:
                config["HostConfig"] = {}
            config["HostConfig"]["DeviceRequests"] = [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]
        
        if self.memory:
            if "HostConfig" not in config:
                config["HostConfig"] = {}
            config["HostConfig"]["Memory"] = int(self.memory.rstrip("m").rstrip("M").rstrip("g").rstrip("G")) * (1024 * 1024 if self.memory.endswith(("m", "M")) else 1024 * 1024 * 1024)
        
        if self.cpus:
            if "HostConfig" not in config:
                config["HostConfig"] = {}
            config["HostConfig"]["NanoCpus"] = int(self.cpus * 1e9)
        
        return config


class Container:
    """Docker container management for Neurenix."""
    
    def __init__(self, container_id: Optional[str] = None):
        """
        Initialize a container.
        
        Args:
            container_id: ID of an existing container to manage
        """
        self.container_id = container_id
        self._check_docker()
    
    def _check_docker(self):
        """Check if Docker is installed and running."""
        try:
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker is not installed or not running")
    
    def create(self, config: ContainerConfig) -> str:
        """
        Create a new container.
        
        Args:
            config: Container configuration
            
        Returns:
            Container ID
        """
        if self.container_id:
            raise RuntimeError("Container already exists")
        
        config_dict = config.to_dict()
        config_json = json.dumps(config_dict)
        
        cmd = ["docker", "create"]
        
        if config.environment:
            for k, v in config.environment.items():
                cmd.extend(["-e", f"{k}={v}"])
        
        if config.volumes:
            for container_path, host_info in config.volumes.items():
                for host_path, mode in host_info.items():
                    cmd.extend(["-v", f"{host_path}:{container_path}:{mode}"])
        
        if config.ports:
            for container_port, host_port in config.ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
        
        if config.working_dir:
            cmd.extend(["-w", config.working_dir])
        
        if config.user:
            cmd.extend(["-u", config.user])
        
        if config.network:
            cmd.extend(["--network", config.network])
        
        if config.hostname:
            cmd.extend(["--hostname", config.hostname])
        
        if config.runtime:
            cmd.extend(["--runtime", config.runtime])
        
        if config.gpu:
            cmd.extend(["--gpus", "all"])
        
        if config.memory:
            cmd.extend(["-m", config.memory])
        
        if config.cpus:
            cmd.extend(["--cpus", str(config.cpus)])
        
        if config.entrypoint:
            if isinstance(config.entrypoint, list):
                cmd.extend(["--entrypoint", config.entrypoint[0]])
                cmd.extend(config.entrypoint[1:])
            else:
                cmd.extend(["--entrypoint", config.entrypoint])
        
        cmd.append(config.image)
        
        if config.command:
            if isinstance(config.command, list):
                cmd.extend(config.command)
            else:
                cmd.extend(config.command.split())
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.container_id = result.stdout.strip()
        return self.container_id
    
    def start(self) -> None:
        """Start the container."""
        if not self.container_id:
            raise RuntimeError("No container to start")
        
        subprocess.run(["docker", "start", self.container_id], check=True)
    
    def stop(self, timeout: int = 10) -> None:
        """
        Stop the container.
        
        Args:
            timeout: Timeout in seconds before killing the container
        """
        if not self.container_id:
            raise RuntimeError("No container to stop")
        
        subprocess.run(["docker", "stop", "-t", str(timeout), self.container_id], check=True)
    
    def remove(self, force: bool = False) -> None:
        """
        Remove the container.
        
        Args:
            force: Whether to force removal of a running container
        """
        if not self.container_id:
            raise RuntimeError("No container to remove")
        
        cmd = ["docker", "rm"]
        if force:
            cmd.append("-f")
        cmd.append(self.container_id)
        
        subprocess.run(cmd, check=True)
        self.container_id = None
    
    def logs(self, follow: bool = False, tail: Optional[str] = None) -> str:
        """
        Get container logs.
        
        Args:
            follow: Whether to follow the logs
            tail: Number of lines to show from the end of the logs
            
        Returns:
            Container logs
        """
        if not self.container_id:
            raise RuntimeError("No container to get logs from")
        
        cmd = ["docker", "logs"]
        if follow:
            cmd.append("-f")
        if tail:
            cmd.extend(["--tail", tail])
        cmd.append(self.container_id)
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    
    def exec(self, command: Union[str, List[str]], interactive: bool = False) -> str:
        """
        Execute a command in the container.
        
        Args:
            command: Command to execute
            interactive: Whether to run in interactive mode
            
        Returns:
            Command output
        """
        if not self.container_id:
            raise RuntimeError("No container to execute command in")
        
        cmd = ["docker", "exec"]
        if interactive:
            cmd.extend(["-i", "-t"])
        cmd.append(self.container_id)
        
        if isinstance(command, list):
            cmd.extend(command)
        else:
            cmd.extend(command.split())
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    
    def inspect(self) -> Dict[str, Any]:
        """
        Inspect the container.
        
        Returns:
            Container information
        """
        if not self.container_id:
            raise RuntimeError("No container to inspect")
        
        result = subprocess.run(["docker", "inspect", self.container_id], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return json.loads(result.stdout)[0]
    
    def wait(self, condition: str = "not-running") -> int:
        """
        Wait for the container to exit.
        
        Args:
            condition: Condition to wait for
            
        Returns:
            Exit code
        """
        if not self.container_id:
            raise RuntimeError("No container to wait for")
        
        result = subprocess.run(["docker", "wait", self.container_id], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return int(result.stdout.strip())
    
    def copy_to(self, src: str, dest: str) -> None:
        """
        Copy a file or directory to the container.
        
        Args:
            src: Source path on the host
            dest: Destination path in the container
        """
        if not self.container_id:
            raise RuntimeError("No container to copy to")
        
        subprocess.run(["docker", "cp", src, f"{self.container_id}:{dest}"], check=True)
    
    def copy_from(self, src: str, dest: str) -> None:
        """
        Copy a file or directory from the container.
        
        Args:
            src: Source path in the container
            dest: Destination path on the host
        """
        if not self.container_id:
            raise RuntimeError("No container to copy from")
        
        subprocess.run(["docker", "cp", f"{self.container_id}:{src}", dest], check=True)
