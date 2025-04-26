"""
Kubernetes service management for Neurenix.

This module provides classes and functions for creating, managing, and
interacting with Kubernetes services for Neurenix models and applications.
"""

import os
import json
import yaml
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union

class ServiceConfig:
    """Configuration for a Kubernetes service."""
    
    def __init__(
        self,
        name: str,
        namespace: str = "default",
        selector: Optional[Dict[str, str]] = None,
        ports: Optional[List[Dict[str, Any]]] = None,
        type: str = "ClusterIP",
        external_ips: Optional[List[str]] = None,
        load_balancer_ip: Optional[str] = None,
        session_affinity: Optional[str] = None,
        external_name: Optional[str] = None,
        external_traffic_policy: Optional[str] = None,
        health_check_node_port: Optional[int] = None,
        publish_not_ready_addresses: bool = False,
        ip_families: Optional[List[str]] = None,
        cluster_ip: Optional[str] = None,
        cluster_ips: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize a service configuration.
        
        Args:
            name: Name of the service
            namespace: Kubernetes namespace
            selector: Label selector for pods
            ports: Service ports
            type: Service type (ClusterIP, NodePort, LoadBalancer, ExternalName)
            external_ips: External IPs for the service
            load_balancer_ip: IP to assign to the load balancer
            session_affinity: Session affinity (None, ClientIP)
            external_name: External name for ExternalName services
            external_traffic_policy: External traffic policy (Local, Cluster)
            health_check_node_port: Health check node port
            publish_not_ready_addresses: Whether to publish not ready addresses
            ip_families: IP families (IPv4, IPv6)
            cluster_ip: Cluster IP
            cluster_ips: Cluster IPs
            labels: Labels to apply to the service
            annotations: Annotations to apply to the service
        """
        self.name = name
        self.namespace = namespace
        self.selector = selector or {"app": name}
        self.ports = ports or [{"port": 80, "targetPort": 80, "protocol": "TCP"}]
        self.type = type
        self.external_ips = external_ips
        self.load_balancer_ip = load_balancer_ip
        self.session_affinity = session_affinity
        self.external_name = external_name
        self.external_traffic_policy = external_traffic_policy
        self.health_check_node_port = health_check_node_port
        self.publish_not_ready_addresses = publish_not_ready_addresses
        self.ip_families = ip_families
        self.cluster_ip = cluster_ip
        self.cluster_ips = cluster_ips
        self.labels = labels or {}
        self.annotations = annotations or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the service
        """
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
                "annotations": self.annotations
            },
            "spec": {
                "selector": self.selector,
                "ports": self.ports,
                "type": self.type
            }
        }
        
        if self.external_ips:
            service["spec"]["externalIPs"] = self.external_ips
        
        if self.load_balancer_ip:
            service["spec"]["loadBalancerIP"] = self.load_balancer_ip
        
        if self.session_affinity:
            service["spec"]["sessionAffinity"] = self.session_affinity
        
        if self.external_name:
            service["spec"]["externalName"] = self.external_name
        
        if self.external_traffic_policy:
            service["spec"]["externalTrafficPolicy"] = self.external_traffic_policy
        
        if self.health_check_node_port:
            service["spec"]["healthCheckNodePort"] = self.health_check_node_port
        
        if self.publish_not_ready_addresses:
            service["spec"]["publishNotReadyAddresses"] = self.publish_not_ready_addresses
        
        if self.ip_families:
            service["spec"]["ipFamilies"] = self.ip_families
        
        if self.cluster_ip:
            service["spec"]["clusterIP"] = self.cluster_ip
        
        if self.cluster_ips:
            service["spec"]["clusterIPs"] = self.cluster_ips
        
        return service
    
    def to_yaml(self) -> str:
        """
        Convert the configuration to YAML.
        
        Returns:
            YAML representation of the service
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)


class Service:
    """Kubernetes service management for Neurenix."""
    
    def __init__(self, name: str, namespace: str = "default"):
        """
        Initialize a service.
        
        Args:
            name: Name of the service
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
    
    def create(self, config: ServiceConfig) -> None:
        """
        Create a service.
        
        Args:
            config: Service configuration
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write(config.to_yaml())
            f.flush()
            subprocess.run(["kubectl", "apply", "-f", f.name], check=True)
    
    def delete(self) -> None:
        """Delete the service."""
        subprocess.run(["kubectl", "delete", "service", self.name, "-n", self.namespace], check=True)
    
    def exists(self) -> bool:
        """
        Check if the service exists.
        
        Returns:
            Whether the service exists
        """
        try:
            subprocess.run(
                ["kubectl", "get", "service", self.name, "-n", self.namespace],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get(self) -> Dict[str, Any]:
        """
        Get the service.
        
        Returns:
            Service information
        """
        result = subprocess.run(
            ["kubectl", "get", "service", self.name, "-n", self.namespace, "-o", "json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return json.loads(result.stdout)
    
    def get_endpoints(self) -> Dict[str, Any]:
        """
        Get the service endpoints.
        
        Returns:
            Service endpoints
        """
        result = subprocess.run(
            ["kubectl", "get", "endpoints", self.name, "-n", self.namespace, "-o", "json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return json.loads(result.stdout)
    
    def get_external_ip(self) -> Optional[str]:
        """
        Get the service external IP.
        
        Returns:
            Service external IP
        """
        service = self.get()
        
        if service["spec"]["type"] == "LoadBalancer":
            if "status" in service and "loadBalancer" in service["status"] and "ingress" in service["status"]["loadBalancer"]:
                ingress = service["status"]["loadBalancer"]["ingress"]
                if ingress and "ip" in ingress[0]:
                    return ingress[0]["ip"]
        
        return None
    
    def get_cluster_ip(self) -> Optional[str]:
        """
        Get the service cluster IP.
        
        Returns:
            Service cluster IP
        """
        service = self.get()
        
        if "spec" in service and "clusterIP" in service["spec"]:
            return service["spec"]["clusterIP"]
        
        return None
    
    def get_node_port(self, port: int) -> Optional[int]:
        """
        Get the service node port.
        
        Args:
            port: Service port
            
        Returns:
            Service node port
        """
        service = self.get()
        
        if service["spec"]["type"] in ["NodePort", "LoadBalancer"]:
            for port_spec in service["spec"]["ports"]:
                if port_spec["port"] == port and "nodePort" in port_spec:
                    return port_spec["nodePort"]
        
        return None
    
    def port_forward(self, local_port: int, remote_port: int) -> subprocess.Popen:
        """
        Forward a port to the service.
        
        Args:
            local_port: Local port
            remote_port: Remote port
            
        Returns:
            Port forwarding process
        """
        return subprocess.Popen(
            ["kubectl", "port-forward", f"service/{self.name}", "-n", self.namespace, f"{local_port}:{remote_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    def create_neurenix_service(
        self,
        port: int = 8000,
        target_port: int = 8000,
        type: str = "ClusterIP",
        selector: Optional[Dict[str, str]] = None,
        external_traffic_policy: Optional[str] = None,
    ) -> None:
        """
        Create a Neurenix service.
        
        Args:
            port: Service port
            target_port: Target port
            type: Service type
            selector: Label selector for pods
            external_traffic_policy: External traffic policy
        """
        config = ServiceConfig(
            name=self.name,
            namespace=self.namespace,
            selector=selector or {"app": self.name, "component": "neurenix"},
            ports=[{"port": port, "targetPort": target_port, "protocol": "TCP"}],
            type=type,
            external_traffic_policy=external_traffic_policy,
            labels={"app": self.name, "component": "neurenix"},
            annotations={"neurenix.ai/service": "true"}
        )
        
        self.create(config)
