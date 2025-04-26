"""
Kubernetes cluster management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Kubernetes clusters for Neurenix models and applications.
"""

import os
import json
import yaml
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

class ClusterConfig:
    """Configuration for a Kubernetes cluster."""
    
    def __init__(
        self,
        name: str,
        provider: str,
        region: str,
        version: str,
        node_pools: List[Dict[str, Any]],
        network: Optional[str] = None,
        subnet: Optional[str] = None,
        private_cluster: bool = False,
        labels: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
        addons: Optional[List[str]] = None,
        auth: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a cluster configuration.
        
        Args:
            name: Name of the cluster
            provider: Cloud provider (gcp, aws, azure)
            region: Cloud region
            version: Kubernetes version
            node_pools: Node pools configuration
            network: Network name
            subnet: Subnet name
            private_cluster: Whether the cluster is private
            labels: Labels to apply to the cluster
            tags: Tags to apply to the cluster
            addons: Addons to enable
            auth: Authentication configuration
        """
        self.name = name
        self.provider = provider
        self.region = region
        self.version = version
        self.node_pools = node_pools
        self.network = network
        self.subnet = subnet
        self.private_cluster = private_cluster
        self.labels = labels or {}
        self.tags = tags or []
        self.addons = addons or []
        self.auth = auth or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the cluster
        """
        config = {
            "name": self.name,
            "provider": self.provider,
            "region": self.region,
            "version": self.version,
            "nodePools": self.node_pools,
            "privateCluster": self.private_cluster,
            "labels": self.labels,
            "tags": self.tags,
            "addons": self.addons,
            "auth": self.auth
        }
        
        if self.network:
            config["network"] = self.network
        
        if self.subnet:
            config["subnet"] = self.subnet
        
        return config
    
    def to_yaml(self) -> str:
        """
        Convert the configuration to YAML.
        
        Returns:
            YAML representation of the cluster
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)


class Cluster:
    """Kubernetes cluster management for Neurenix."""
    
    def __init__(self, name: str, provider: str = "gcp"):
        """
        Initialize a cluster.
        
        Args:
            name: Name of the cluster
            provider: Cloud provider (gcp, aws, azure)
        """
        self.name = name
        self.provider = provider
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if the required dependencies are installed."""
        try:
            subprocess.run(["kubectl", "version", "--client"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("kubectl is not installed or not configured")
        
        if self.provider == "gcp":
            try:
                subprocess.run(["gcloud", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("gcloud is not installed or not configured")
        elif self.provider == "aws":
            try:
                subprocess.run(["aws", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("aws CLI is not installed or not configured")
        elif self.provider == "azure":
            try:
                subprocess.run(["az", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("azure CLI is not installed or not configured")
    
    def create(self, config: ClusterConfig) -> None:
        """
        Create a cluster.
        
        Args:
            config: Cluster configuration
        """
        if self.provider == "gcp":
            self._create_gcp_cluster(config)
        elif self.provider == "aws":
            self._create_aws_cluster(config)
        elif self.provider == "azure":
            self._create_azure_cluster(config)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _create_gcp_cluster(self, config: ClusterConfig) -> None:
        """
        Create a GCP GKE cluster.
        
        Args:
            config: Cluster configuration
        """
        cmd = [
            "gcloud", "container", "clusters", "create", config.name,
            "--region", config.region,
            "--cluster-version", config.version
        ]
        
        if config.network:
            cmd.extend(["--network", config.network])
        
        if config.subnet:
            cmd.extend(["--subnetwork", config.subnet])
        
        if config.private_cluster:
            cmd.append("--enable-private-nodes")
            cmd.append("--enable-private-endpoint")
            cmd.append("--master-ipv4-cidr=172.16.0.0/28")
        
        for key, value in config.labels.items():
            cmd.extend(["--labels", f"{key}={value}"])
        
        for tag in config.tags:
            cmd.extend(["--tags", tag])
        
        for addon in config.addons:
            cmd.extend(["--addons", addon])
        
        subprocess.run(cmd, check=True)
        
        for node_pool in config.node_pools:
            self._create_gcp_node_pool(node_pool)
    
    def _create_gcp_node_pool(self, node_pool: Dict[str, Any]) -> None:
        """
        Create a GCP GKE node pool.
        
        Args:
            node_pool: Node pool configuration
        """
        cmd = [
            "gcloud", "container", "node-pools", "create", node_pool["name"],
            "--cluster", self.name,
            "--region", node_pool.get("region", "us-central1"),
            "--machine-type", node_pool.get("machineType", "e2-standard-4"),
            "--num-nodes", str(node_pool.get("numNodes", 3))
        ]
        
        if "diskSizeGb" in node_pool:
            cmd.extend(["--disk-size", str(node_pool["diskSizeGb"])])
        
        if "diskType" in node_pool:
            cmd.extend(["--disk-type", node_pool["diskType"]])
        
        if "accelerator" in node_pool:
            cmd.extend([
                "--accelerator",
                f"type={node_pool['accelerator']['type']},count={node_pool['accelerator']['count']}"
            ])
        
        if "labels" in node_pool:
            for key, value in node_pool["labels"].items():
                cmd.extend(["--node-labels", f"{key}={value}"])
        
        if "taints" in node_pool:
            for taint in node_pool["taints"]:
                cmd.extend(["--node-taints", f"{taint['key']}={taint['value']}:{taint['effect']}"])
        
        if "preemptible" in node_pool and node_pool["preemptible"]:
            cmd.append("--preemptible")
        
        if "autoscaling" in node_pool and node_pool["autoscaling"]:
            cmd.extend([
                "--enable-autoscaling",
                f"--min-nodes={node_pool['autoscaling']['minNodes']}",
                f"--max-nodes={node_pool['autoscaling']['maxNodes']}"
            ])
        
        subprocess.run(cmd, check=True)
    
    def _create_aws_cluster(self, config: ClusterConfig) -> None:
        """
        Create an AWS EKS cluster.
        
        Args:
            config: Cluster configuration
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            cluster_config = {
                "apiVersion": "eksctl.io/v1alpha5",
                "kind": "ClusterConfig",
                "metadata": {
                    "name": config.name,
                    "region": config.region,
                    "version": config.version,
                    "tags": {key: value for key, value in config.labels.items()}
                },
                "vpc": {
                    "id": config.network,
                    "subnets": {
                        "private": {
                            "id": config.subnet
                        }
                    }
                },
                "nodeGroups": []
            }
            
            for node_pool in config.node_pools:
                node_group = {
                    "name": node_pool["name"],
                    "instanceType": node_pool.get("machineType", "m5.large"),
                    "desiredCapacity": node_pool.get("numNodes", 3),
                    "volumeSize": node_pool.get("diskSizeGb", 80),
                    "volumeType": node_pool.get("diskType", "gp2"),
                    "labels": node_pool.get("labels", {}),
                    "tags": node_pool.get("tags", {})
                }
                
                if "taints" in node_pool:
                    node_group["taints"] = node_pool["taints"]
                
                if "autoscaling" in node_pool and node_pool["autoscaling"]:
                    node_group["minSize"] = node_pool["autoscaling"]["minNodes"]
                    node_group["maxSize"] = node_pool["autoscaling"]["maxNodes"]
                
                cluster_config["nodeGroups"].append(node_group)
            
            f.write(yaml.dump(cluster_config, default_flow_style=False))
            f.flush()
            
            subprocess.run(["eksctl", "create", "cluster", "-f", f.name], check=True)
    
    def _create_azure_cluster(self, config: ClusterConfig) -> None:
        """
        Create an Azure AKS cluster.
        
        Args:
            config: Cluster configuration
        """
        cmd = [
            "az", "aks", "create",
            "--resource-group", config.auth.get("resourceGroup", "neurenix-rg"),
            "--name", config.name,
            "--location", config.region,
            "--kubernetes-version", config.version,
            "--node-count", str(config.node_pools[0].get("numNodes", 3)),
            "--node-vm-size", config.node_pools[0].get("machineType", "Standard_DS2_v2")
        ]
        
        if config.network:
            cmd.extend(["--vnet-subnet-id", config.subnet])
        
        for key, value in config.labels.items():
            cmd.extend(["--tags", f"{key}={value}"])
        
        if "enablePrivateCluster" in config.auth and config.auth["enablePrivateCluster"]:
            cmd.append("--enable-private-cluster")
        
        subprocess.run(cmd, check=True)
        
        for node_pool in config.node_pools[1:]:
            self._create_azure_node_pool(node_pool, config.auth.get("resourceGroup", "neurenix-rg"))
    
    def _create_azure_node_pool(self, node_pool: Dict[str, Any], resource_group: str) -> None:
        """
        Create an Azure AKS node pool.
        
        Args:
            node_pool: Node pool configuration
            resource_group: Azure resource group
        """
        cmd = [
            "az", "aks", "nodepool", "add",
            "--resource-group", resource_group,
            "--cluster-name", self.name,
            "--name", node_pool["name"],
            "--node-count", str(node_pool.get("numNodes", 3)),
            "--node-vm-size", node_pool.get("machineType", "Standard_DS2_v2")
        ]
        
        if "labels" in node_pool:
            labels_str = " ".join([f"{k}={v}" for k, v in node_pool["labels"].items()])
            cmd.extend(["--labels", labels_str])
        
        if "taints" in node_pool:
            taints_str = " ".join([f"{t['key']}={t['value']}:{t['effect']}" for t in node_pool["taints"]])
            cmd.extend(["--node-taints", taints_str])
        
        if "autoscaling" in node_pool and node_pool["autoscaling"]:
            cmd.extend([
                "--enable-cluster-autoscaler",
                "--min-count", str(node_pool["autoscaling"]["minNodes"]),
                "--max-count", str(node_pool["autoscaling"]["maxNodes"])
            ])
        
        subprocess.run(cmd, check=True)
    
    def delete(self) -> None:
        """Delete the cluster."""
        if self.provider == "gcp":
            subprocess.run(["gcloud", "container", "clusters", "delete", self.name, "--quiet"], check=True)
        elif self.provider == "aws":
            subprocess.run(["eksctl", "delete", "cluster", "--name", self.name], check=True)
        elif self.provider == "azure":
            subprocess.run(["az", "aks", "delete", "--name", self.name, "--resource-group", "neurenix-rg", "--yes"], check=True)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def exists(self) -> bool:
        """
        Check if the cluster exists.
        
        Returns:
            Whether the cluster exists
        """
        try:
            if self.provider == "gcp":
                subprocess.run(
                    ["gcloud", "container", "clusters", "describe", self.name],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            elif self.provider == "aws":
                subprocess.run(
                    ["aws", "eks", "describe-cluster", "--name", self.name],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            elif self.provider == "azure":
                subprocess.run(
                    ["az", "aks", "show", "--name", self.name, "--resource-group", "neurenix-rg"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get(self) -> Dict[str, Any]:
        """
        Get the cluster.
        
        Returns:
            Cluster information
        """
        if self.provider == "gcp":
            result = subprocess.run(
                ["gcloud", "container", "clusters", "describe", self.name, "--format", "json"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        elif self.provider == "aws":
            result = subprocess.run(
                ["aws", "eks", "describe-cluster", "--name", self.name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        elif self.provider == "azure":
            result = subprocess.run(
                ["az", "aks", "show", "--name", self.name, "--resource-group", "neurenix-rg"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        return json.loads(result.stdout)
    
    def get_credentials(self) -> None:
        """Get the cluster credentials."""
        if self.provider == "gcp":
            subprocess.run(["gcloud", "container", "clusters", "get-credentials", self.name], check=True)
        elif self.provider == "aws":
            subprocess.run(["aws", "eks", "update-kubeconfig", "--name", self.name], check=True)
        elif self.provider == "azure":
            subprocess.run(["az", "aks", "get-credentials", "--name", self.name, "--resource-group", "neurenix-rg"], check=True)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def create_neurenix_cluster(
        self,
        region: str,
        version: str,
        node_count: int = 3,
        machine_type: str = "n1-standard-4",
        gpu_node_count: int = 0,
        gpu_type: str = "nvidia-tesla-t4",
        private_cluster: bool = False,
        network: Optional[str] = None,
        subnet: Optional[str] = None,
    ) -> None:
        """
        Create a Neurenix cluster.
        
        Args:
            region: Cloud region
            version: Kubernetes version
            node_count: Number of nodes
            machine_type: Machine type
            gpu_node_count: Number of GPU nodes
            gpu_type: GPU type
            private_cluster: Whether the cluster is private
            network: Network name
            subnet: Subnet name
        """
        node_pools = [
            {
                "name": "default-pool",
                "machineType": machine_type,
                "numNodes": node_count,
                "diskSizeGb": 100,
                "diskType": "pd-standard",
                "labels": {
                    "neurenix.ai/node-type": "cpu"
                },
                "autoscaling": {
                    "enabled": True,
                    "minNodes": 1,
                    "maxNodes": node_count * 2
                }
            }
        ]
        
        if gpu_node_count > 0:
            node_pools.append({
                "name": "gpu-pool",
                "machineType": "n1-standard-8",
                "numNodes": gpu_node_count,
                "diskSizeGb": 200,
                "diskType": "pd-ssd",
                "accelerator": {
                    "type": gpu_type,
                    "count": 1
                },
                "labels": {
                    "neurenix.ai/node-type": "gpu",
                    "neurenix.ai/gpu-type": gpu_type
                },
                "taints": [
                    {
                        "key": "nvidia.com/gpu",
                        "value": "present",
                        "effect": "NoSchedule"
                    }
                ],
                "autoscaling": {
                    "enabled": True,
                    "minNodes": 0,
                    "maxNodes": gpu_node_count * 2
                }
            })
        
        config = ClusterConfig(
            name=self.name,
            provider=self.provider,
            region=region,
            version=version,
            node_pools=node_pools,
            network=network,
            subnet=subnet,
            private_cluster=private_cluster,
            labels={
                "neurenix.ai/cluster": "true",
                "neurenix.ai/version": "1.0"
            },
            addons=["HorizontalPodAutoscaling", "HttpLoadBalancing", "NetworkPolicy"]
        )
        
        self.create(config)
