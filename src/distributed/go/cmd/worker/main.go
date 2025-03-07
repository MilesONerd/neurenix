// Package main provides the worker node implementation for Neurenix distributed computing.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/MilesONerd/framework/src/distributed/go/pkg/worker"
)

var (
	id = flag.String("id", "", "Worker ID")
	address = flag.String("address", ":50052", "Address to listen on")
	coordinatorAddress = flag.String("coordinator", "localhost:50051", "Coordinator address")
	gpuCount = flag.Int("gpus", 0, "Number of GPUs available")
	totalRAM = flag.Int64("ram", 0, "Total RAM available in bytes")
)

func main() {
	flag.Parse()

	if *id == "" {
		log.Fatal("Worker ID is required")
	}

	log.Printf("Starting Neurenix worker node %s", *id)

	// Create worker
	w := worker.NewWorker(*id, *address, *coordinatorAddress, *gpuCount, *totalRAM)

	// Register task handlers
	// This is a placeholder implementation
	// Real implementation would register task handlers for different task types
	w.RegisterTaskHandler("training", func(ctx context.Context, taskData map[string]interface{}) (map[string]interface{}, error) {
		log.Printf("Executing training task with data: %v", taskData)
		// Placeholder implementation
		return map[string]interface{}{"status": "success"}, nil
	})

	w.RegisterTaskHandler("inference", func(ctx context.Context, taskData map[string]interface{}) (map[string]interface{}, error) {
		log.Printf("Executing inference task with data: %v", taskData)
		// Placeholder implementation
		return map[string]interface{}{"status": "success"}, nil
	})

	// Start worker
	if err := w.Start(); err != nil {
		log.Fatalf("Failed to start worker: %v", err)
	}

	// Handle signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Wait for signal
	sig := <-sigCh
	log.Printf("Received signal %v, shutting down", sig)

	// Stop worker
	if err := w.Stop(); err != nil {
		log.Printf("Failed to stop worker: %v", err)
	}

	log.Printf("Neurenix worker node %s stopped", *id)
}
