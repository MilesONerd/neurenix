"""
Docker image management for Neurenix.

This module provides classes and functions for building, managing, and
interacting with Docker images for Neurenix models and applications.
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, List, Optional, Union, Any

class ImageBuilder:
    """Builder for Docker images."""
    
    def __init__(self, path: Optional[str] = None):
        """
        Initialize an image builder.
        
        Args:
            path: Path to a directory containing a Dockerfile
        """
        self.path = path
        self._check_docker()
    
    def _check_docker(self):
        """Check if Docker is installed and running."""
        try:
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker is not installed or not running")
    
    def build(
        self,
        tag: str,
        dockerfile: Optional[str] = None,
        context: Optional[str] = None,
        build_args: Optional[Dict[str, str]] = None,
        no_cache: bool = False,
        pull: bool = False,
        platform: Optional[str] = None,
        target: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build a Docker image.
        
        Args:
            tag: Tag for the image
            dockerfile: Path to a Dockerfile (overrides path)
            context: Path to the build context (defaults to path)
            build_args: Build arguments
            no_cache: Whether to use cache
            pull: Whether to pull base images
            platform: Platform to build for
            target: Target build stage
            labels: Labels to apply to the image
            
        Returns:
            Image ID
        """
        if not self.path and not context:
            raise ValueError("Either path or context must be provided")
        
        context_path = context or self.path
        dockerfile_path = dockerfile or (os.path.join(self.path, "Dockerfile") if self.path else None)
        
        cmd = ["docker", "build"]
        
        if tag:
            cmd.extend(["-t", tag])
        
        if dockerfile_path:
            cmd.extend(["-f", dockerfile_path])
        
        if build_args:
            for k, v in build_args.items():
                cmd.extend(["--build-arg", f"{k}={v}"])
        
        if no_cache:
            cmd.append("--no-cache")
        
        if pull:
            cmd.append("--pull")
        
        if platform:
            cmd.extend(["--platform", platform])
        
        if target:
            cmd.extend(["--target", target])
        
        if labels:
            for k, v in labels.items():
                cmd.extend(["--label", f"{k}={v}"])
        
        cmd.append(context_path)
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in result.stdout.splitlines():
            if "Successfully built" in line:
                return line.split()[-1]
        
        return ""
    
    def create_dockerfile(
        self,
        base_image: str,
        commands: List[str],
        output_path: Optional[str] = None,
        entrypoint: Optional[Union[str, List[str]]] = None,
        cmd: Optional[Union[str, List[str]]] = None,
        env: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        expose: Optional[List[str]] = None,
        volumes: Optional[List[str]] = None,
        workdir: Optional[str] = None,
        user: Optional[str] = None,
    ) -> str:
        """
        Create a Dockerfile.
        
        Args:
            base_image: Base image to use
            commands: Commands to run in the Dockerfile
            output_path: Path to write the Dockerfile to
            entrypoint: Entrypoint for the container
            cmd: Command for the container
            env: Environment variables
            labels: Labels to apply to the image
            expose: Ports to expose
            volumes: Volumes to create
            workdir: Working directory
            user: User to run as
            
        Returns:
            Path to the created Dockerfile
        """
        dockerfile_content = [f"FROM {base_image}"]
        
        if env:
            for k, v in env.items():
                dockerfile_content.append(f"ENV {k}={v}")
        
        if labels:
            for k, v in labels.items():
                dockerfile_content.append(f"LABEL {k}={v}")
        
        if workdir:
            dockerfile_content.append(f"WORKDIR {workdir}")
        
        if user:
            dockerfile_content.append(f"USER {user}")
        
        for command in commands:
            dockerfile_content.append(f"RUN {command}")
        
        if expose:
            for port in expose:
                dockerfile_content.append(f"EXPOSE {port}")
        
        if volumes:
            for volume in volumes:
                dockerfile_content.append(f"VOLUME {volume}")
        
        if entrypoint:
            if isinstance(entrypoint, list):
                entrypoint_str = json.dumps(entrypoint)
            else:
                entrypoint_str = f'["{entrypoint}"]'
            dockerfile_content.append(f"ENTRYPOINT {entrypoint_str}")
        
        if cmd:
            if isinstance(cmd, list):
                cmd_str = json.dumps(cmd)
            else:
                cmd_str = f'["{cmd}"]'
            dockerfile_content.append(f"CMD {cmd_str}")
        
        dockerfile_content = "\n".join(dockerfile_content)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(dockerfile_content)
            return output_path
        else:
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "Dockerfile")
            with open(temp_file, "w") as f:
                f.write(dockerfile_content)
            return temp_file
    
    def create_neurenix_dockerfile(
        self,
        output_path: Optional[str] = None,
        cuda: bool = False,
        python_version: str = "3.9",
        neurenix_version: Optional[str] = None,
        additional_packages: Optional[List[str]] = None,
        entrypoint: Optional[Union[str, List[str]]] = None,
        cmd: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """
        Create a Dockerfile for a Neurenix application.
        
        Args:
            output_path: Path to write the Dockerfile to
            cuda: Whether to use CUDA
            python_version: Python version to use
            neurenix_version: Neurenix version to install
            additional_packages: Additional packages to install
            entrypoint: Entrypoint for the container
            cmd: Command for the container
            
        Returns:
            Path to the created Dockerfile
        """
        base_image = f"python:{python_version}-slim" if not cuda else f"nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04"
        
        commands = []
        
        if cuda:
            commands.extend([
                "apt-get update && apt-get install -y --no-install-recommends python3-pip python3-dev && rm -rf /var/lib/apt/lists/*",
                f"ln -sf /usr/bin/python3 /usr/bin/python && pip3 install --no-cache-dir --upgrade pip"
            ])
        
        commands.append("pip install --no-cache-dir numpy")
        
        if neurenix_version:
            commands.append(f"pip install --no-cache-dir neurenix=={neurenix_version}")
        else:
            commands.append("pip install --no-cache-dir neurenix")
        
        if additional_packages:
            commands.append(f"pip install --no-cache-dir {' '.join(additional_packages)}")
        
        return self.create_dockerfile(
            base_image=base_image,
            commands=commands,
            output_path=output_path,
            entrypoint=entrypoint,
            cmd=cmd,
            workdir="/app",
            env={"PYTHONUNBUFFERED": "1"}
        )


class Image:
    """Docker image management for Neurenix."""
    
    def __init__(self, name: str):
        """
        Initialize an image.
        
        Args:
            name: Name of the image
        """
        self.name = name
        self._check_docker()
    
    def _check_docker(self):
        """Check if Docker is installed and running."""
        try:
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker is not installed or not running")
    
    def exists(self) -> bool:
        """
        Check if the image exists.
        
        Returns:
            Whether the image exists
        """
        try:
            subprocess.run(["docker", "image", "inspect", self.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def pull(self, platform: Optional[str] = None) -> None:
        """
        Pull the image.
        
        Args:
            platform: Platform to pull for
        """
        cmd = ["docker", "pull"]
        
        if platform:
            cmd.extend(["--platform", platform])
        
        cmd.append(self.name)
        
        subprocess.run(cmd, check=True)
    
    def push(self) -> None:
        """Push the image."""
        subprocess.run(["docker", "push", self.name], check=True)
    
    def tag(self, new_name: str) -> None:
        """
        Tag the image.
        
        Args:
            new_name: New name for the image
        """
        subprocess.run(["docker", "tag", self.name, new_name], check=True)
    
    def remove(self, force: bool = False) -> None:
        """
        Remove the image.
        
        Args:
            force: Whether to force removal
        """
        cmd = ["docker", "rmi"]
        
        if force:
            cmd.append("-f")
        
        cmd.append(self.name)
        
        subprocess.run(cmd, check=True)
    
    def inspect(self) -> Dict[str, Any]:
        """
        Inspect the image.
        
        Returns:
            Image information
        """
        result = subprocess.run(["docker", "image", "inspect", self.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return json.loads(result.stdout)[0]
    
    def history(self) -> List[Dict[str, Any]]:
        """
        Get the image history.
        
        Returns:
            Image history
        """
        result = subprocess.run(["docker", "image", "history", "--format", "{{json .}}", self.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    
    def save(self, output_path: str) -> None:
        """
        Save the image to a file.
        
        Args:
            output_path: Path to save the image to
        """
        subprocess.run(["docker", "save", "-o", output_path, self.name], check=True)
    
    def load(self, input_path: str) -> None:
        """
        Load the image from a file.
        
        Args:
            input_path: Path to load the image from
        """
        subprocess.run(["docker", "load", "-i", input_path], check=True)
