// Package rpc provides RPC functionality for distributed communication in Neurenix.
package rpc

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// RPCServer represents an RPC server.
type RPCServer struct {
	server   *grpc.Server
	listener net.Listener
	address  string
	mu       sync.Mutex
	running  bool
}

// NewRPCServer creates a new RPC server.
func NewRPCServer(address string) *RPCServer {
	return &RPCServer{
		address: address,
	}
}

// Start starts the RPC server.
func (s *RPCServer) Start() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return errors.New("server already running")
	}

	listener, err := net.Listen("tcp", s.address)
	if err != nil {
		return fmt.Errorf("failed to listen: %v", err)
	}

	s.listener = listener
	s.server = grpc.NewServer()
	s.running = true

	log.Printf("RPC server started on %s", s.address)

	go func() {
		if err := s.server.Serve(listener); err != nil {
			log.Printf("Failed to serve: %v", err)
		}
	}()

	return nil
}

// Stop stops the RPC server.
func (s *RPCServer) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return errors.New("server not running")
	}

	s.server.GracefulStop()
	s.running = false

	log.Printf("RPC server stopped")
	return nil
}

// RegisterService registers a service with the RPC server.
func (s *RPCServer) RegisterService(desc *grpc.ServiceDesc, impl interface{}) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.server == nil {
		return errors.New("server not initialized")
	}

	s.server.RegisterService(desc, impl)
	return nil
}

// RPCClient represents an RPC client.
type RPCClient struct {
	conn    *grpc.ClientConn
	address string
	mu      sync.Mutex
	connected bool
}

// NewRPCClient creates a new RPC client.
func NewRPCClient(address string) *RPCClient {
	return &RPCClient{
		address: address,
	}
}

// Connect connects the RPC client to the server.
func (c *RPCClient) Connect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.connected {
		return errors.New("client already connected")
	}

	conn, err := grpc.DialContext(
		ctx,
		c.address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return fmt.Errorf("failed to connect: %v", err)
	}

	c.conn = conn
	c.connected = true

	log.Printf("RPC client connected to %s", c.address)
	return nil
}

// Disconnect disconnects the RPC client from the server.
func (c *RPCClient) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.connected {
		return errors.New("client not connected")
	}

	if err := c.conn.Close(); err != nil {
		return fmt.Errorf("failed to close connection: %v", err)
	}

	c.connected = false
	log.Printf("RPC client disconnected from %s", c.address)
	return nil
}

// GetConnection gets the gRPC client connection.
func (c *RPCClient) GetConnection() (*grpc.ClientConn, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.connected {
		return nil, errors.New("client not connected")
	}

	return c.conn, nil
}

// RPCPool represents a pool of RPC clients.
type RPCPool struct {
	clients map[string]*RPCClient
	mu      sync.RWMutex
}

// NewRPCPool creates a new RPC pool.
func NewRPCPool() *RPCPool {
	return &RPCPool{
		clients: make(map[string]*RPCClient),
	}
}

// GetClient gets a client from the pool.
func (p *RPCPool) GetClient(address string) (*RPCClient, error) {
	p.mu.RLock()
	client, exists := p.clients[address]
	p.mu.RUnlock()

	if exists {
		return client, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// Check again in case another goroutine created the client
	client, exists = p.clients[address]
	if exists {
		return client, nil
	}

	client = NewRPCClient(address)
	p.clients[address] = client

	return client, nil
}

// ConnectAll connects all clients in the pool.
func (p *RPCPool) ConnectAll(ctx context.Context) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	for address, client := range p.clients {
		if err := client.Connect(ctx); err != nil {
			return fmt.Errorf("failed to connect to %s: %v", address, err)
		}
	}

	return nil
}

// DisconnectAll disconnects all clients in the pool.
func (p *RPCPool) DisconnectAll() error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	for address, client := range p.clients {
		if err := client.Disconnect(); err != nil {
			return fmt.Errorf("failed to disconnect from %s: %v", address, err)
		}
	}

	return nil
}

// RemoveClient removes a client from the pool.
func (p *RPCPool) RemoveClient(address string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	client, exists := p.clients[address]
	if !exists {
		return fmt.Errorf("client for %s not found", address)
	}

	if err := client.Disconnect(); err != nil {
		return fmt.Errorf("failed to disconnect from %s: %v", address, err)
	}

	delete(p.clients, address)
	return nil
}
