// Package cluster provides functionality for managing distributed clusters in Neurenix.
package cluster

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"
)

// NodeStatus represents the status of a node in the cluster.
type NodeStatus int

const (
	// NodeStatusUnknown indicates that the node status is unknown.
	NodeStatusUnknown NodeStatus = iota
	// NodeStatusOnline indicates that the node is online and available.
	NodeStatusOnline
	// NodeStatusOffline indicates that the node is offline.
	NodeStatusOffline
	// NodeStatusBusy indicates that the node is busy with a task.
	NodeStatusBusy
	// NodeStatusError indicates that the node is in an error state.
	NodeStatusError
)

// NodeType represents the type of a node in the cluster.
type NodeType int

const (
	// NodeTypeCoordinator indicates that the node is a coordinator.
	NodeTypeCoordinator NodeType = iota
	// NodeTypeWorker indicates that the node is a worker.
	NodeTypeWorker
)

// NodeInfo represents information about a node in the cluster.
type NodeInfo struct {
	ID         string
	Type       NodeType
	Status     NodeStatus
	Address    string
	GPUCount   int
	TotalRAM   int64
	AvailRAM   int64
	LastSeen   time.Time
	Tags       map[string]string
	Attributes map[string]interface{}
}

// Cluster represents a distributed cluster of nodes.
type Cluster struct {
	mu    sync.RWMutex
	nodes map[string]*NodeInfo
}

// NewCluster creates a new cluster.
func NewCluster() *Cluster {
	return &Cluster{
		nodes: make(map[string]*NodeInfo),
	}
}

// RegisterNode registers a node with the cluster.
func (c *Cluster) RegisterNode(node *NodeInfo) error {
	if node == nil {
		return errors.New("node cannot be nil")
	}
	if node.ID == "" {
		return errors.New("node ID cannot be empty")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	node.LastSeen = time.Now()
	c.nodes[node.ID] = node

	log.Printf("Node %s registered with the cluster", node.ID)
	return nil
}

// UnregisterNode unregisters a node from the cluster.
func (c *Cluster) UnregisterNode(nodeID string) error {
	if nodeID == "" {
		return errors.New("node ID cannot be empty")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.nodes[nodeID]; !exists {
		return fmt.Errorf("node %s not found", nodeID)
	}

	delete(c.nodes, nodeID)
	log.Printf("Node %s unregistered from the cluster", nodeID)
	return nil
}

// GetNode gets a node from the cluster.
func (c *Cluster) GetNode(nodeID string) (*NodeInfo, error) {
	if nodeID == "" {
		return nil, errors.New("node ID cannot be empty")
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	node, exists := c.nodes[nodeID]
	if !exists {
		return nil, fmt.Errorf("node %s not found", nodeID)
	}

	return node, nil
}

// UpdateNodeStatus updates the status of a node in the cluster.
func (c *Cluster) UpdateNodeStatus(nodeID string, status NodeStatus) error {
	if nodeID == "" {
		return errors.New("node ID cannot be empty")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	node, exists := c.nodes[nodeID]
	if !exists {
		return fmt.Errorf("node %s not found", nodeID)
	}

	node.Status = status
	node.LastSeen = time.Now()
	return nil
}

// GetAllNodes gets all nodes in the cluster.
func (c *Cluster) GetAllNodes() []*NodeInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	nodes := make([]*NodeInfo, 0, len(c.nodes))
	for _, node := range c.nodes {
		nodes = append(nodes, node)
	}

	return nodes
}

// GetNodesByType gets all nodes of a specific type in the cluster.
func (c *Cluster) GetNodesByType(nodeType NodeType) []*NodeInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	nodes := make([]*NodeInfo, 0)
	for _, node := range c.nodes {
		if node.Type == nodeType {
			nodes = append(nodes, node)
		}
	}

	return nodes
}

// GetNodesByStatus gets all nodes with a specific status in the cluster.
func (c *Cluster) GetNodesByStatus(status NodeStatus) []*NodeInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	nodes := make([]*NodeInfo, 0)
	for _, node := range c.nodes {
		if node.Status == status {
			nodes = append(nodes, node)
		}
	}

	return nodes
}

// GetAvailableWorkers gets all available worker nodes in the cluster.
func (c *Cluster) GetAvailableWorkers() []*NodeInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	workers := make([]*NodeInfo, 0)
	for _, node := range c.nodes {
		if node.Type == NodeTypeWorker && node.Status == NodeStatusOnline {
			workers = append(workers, node)
		}
	}

	return workers
}

// MonitorCluster starts monitoring the cluster for node health.
func (c *Cluster) MonitorCluster(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.checkNodeHealth()
		}
	}
}

// checkNodeHealth checks the health of all nodes in the cluster.
func (c *Cluster) checkNodeHealth() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for id, node := range c.nodes {
		// If a node hasn't been seen in 1 minute, mark it as offline
		if now.Sub(node.LastSeen) > time.Minute {
			log.Printf("Node %s hasn't been seen in over a minute, marking as offline", id)
			node.Status = NodeStatusOffline
		}
	}
}

// String returns a string representation of the cluster.
func (c *Cluster) String() string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return fmt.Sprintf("Cluster with %d nodes", len(c.nodes))
}
