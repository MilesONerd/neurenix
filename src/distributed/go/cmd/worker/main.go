// Package main provides the worker node implementation for Neurenix distributed computing.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/MilesONerd/neurenix/src/distributed/go/pkg/worker"
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

	w.RegisterTaskHandler("training", func(ctx context.Context, taskData map[string]interface{}) (map[string]interface{}, error) {
		log.Printf("Executing training task with data: %v", taskData)
		
		modelType, ok := taskData["model_type"].(string)
		if !ok {
			return nil, fmt.Errorf("model_type not provided or not a string")
		}
		
		epochs, ok := taskData["epochs"].(float64)
		if !ok {
			epochs = 10 // Default value
		}
		
		batchSize, ok := taskData["batch_size"].(float64)
		if !ok {
			batchSize = 32 // Default value
		}
		
		log.Printf("Starting training of %s model for %v epochs with batch size %v", modelType, epochs, batchSize)
		
		for i := 0; i < int(epochs); i++ {
			select {
			case <-ctx.Done():
				log.Printf("Training cancelled at epoch %d", i)
				return map[string]interface{}{
					"status":       "cancelled",
					"completed_epochs": i,
				}, nil
			default:
				time.Sleep(100 * time.Millisecond)
				log.Printf("Completed epoch %d", i+1)
			}
		}
		
		return map[string]interface{}{
			"status":       "success",
			"model_type":   modelType,
			"epochs":       epochs,
			"batch_size":   batchSize,
			"accuracy":     0.95, // Simulated accuracy
			"loss":         0.05, // Simulated loss
			"training_time": epochs * batchSize * 0.01, // Simulated training time
		}, nil
	})

	w.RegisterTaskHandler("inference", func(ctx context.Context, taskData map[string]interface{}) (map[string]interface{}, error) {
		log.Printf("Executing inference task with data: %v", taskData)
		
		modelType, ok := taskData["model_type"].(string)
		if !ok {
			return nil, fmt.Errorf("model_type not provided or not a string")
		}
		
		inputData, ok := taskData["input_data"].(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("input_data not provided or not a map")
		}
		
		log.Printf("Running inference with %s model", modelType)
		
		select {
		case <-ctx.Done():
			log.Printf("Inference cancelled")
			return map[string]interface{}{
				"status": "cancelled",
			}, nil
		default:
			time.Sleep(200 * time.Millisecond)
		}
		
		var results map[string]interface{}
		switch modelType {
		case "classification":
			results = map[string]interface{}{
				"class": "class_1",
				"confidence": 0.92,
				"probabilities": map[string]float64{
					"class_1": 0.92,
					"class_2": 0.05,
					"class_3": 0.03,
				},
			}
		case "regression":
			results = map[string]interface{}{
				"prediction": 42.5,
				"confidence_interval": map[string]float64{
					"lower": 40.2,
					"upper": 44.8,
				},
			}
		default:
			results = map[string]interface{}{
				"prediction": "unknown",
			}
		}
		
		return map[string]interface{}{
			"status":       "success",
			"model_type":   modelType,
			"results":      results,
			"inference_time": 0.2, // Simulated inference time in seconds
		}, nil
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
