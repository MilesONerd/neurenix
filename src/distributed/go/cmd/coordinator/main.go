// Package main provides the coordinator node implementation for Neurenix distributed computing.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/MilesONerd/framework/src/distributed/go/pkg/cluster"
	"github.com/MilesONerd/framework/src/distributed/go/pkg/rpc"
	"github.com/MilesONerd/framework/src/distributed/go/pkg/scheduler"
)

var (
	address = flag.String("address", ":50051", "Address to listen on")
)

func main() {
	flag.Parse()

	log.Println("Starting Neurenix coordinator node")

	// Create cluster
	c := cluster.NewCluster()

	// Create scheduler
	s := scheduler.NewScheduler(c)
	if err := s.Start(); err != nil {
		log.Fatalf("Failed to start scheduler: %v", err)
	}

	// Create RPC server
	server := rpc.NewRPCServer(*address)
	if err := server.Start(); err != nil {
		log.Fatalf("Failed to start RPC server: %v", err)
	}

	// Register services
	// This is a placeholder implementation
	// Real implementation would register gRPC services

	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start cluster monitoring
	go c.MonitorCluster(ctx, 30*time.Second)

	// Handle signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Wait for signal
	sig := <-sigCh
	log.Printf("Received signal %v, shutting down", sig)

	// Stop scheduler
	if err := s.Stop(); err != nil {
		log.Printf("Failed to stop scheduler: %v", err)
	}

	// Stop RPC server
	if err := server.Stop(); err != nil {
		log.Printf("Failed to stop RPC server: %v", err)
	}

	log.Println("Neurenix coordinator node stopped")
}
