// Package worker provides worker functionality for distributed computing in Neurenix.
package worker

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/MilesONerd/neurenix/src/distributed/go/pkg/cluster"
	"github.com/MilesONerd/neurenix/src/distributed/go/pkg/rpc"
)

// TaskHandler represents a handler for a specific task type.
type TaskHandler func(ctx context.Context, taskData map[string]interface{}) (map[string]interface{}, error)

// Worker represents a worker node in a distributed cluster.
type Worker struct {
	mu            sync.RWMutex
	id            string
	address       string
	coordinatorAddress string
	rpcServer     *rpc.RPCServer
	rpcClient     *rpc.RPCClient
	taskHandlers  map[string]TaskHandler
	gpuCount      int
	totalRAM      int64
	availRAM      int64
	status        cluster.NodeStatus
	ctx           context.Context
	cancel        context.CancelFunc
	runningTasks  map[string]context.CancelFunc
}

// NewWorker creates a new worker.
func NewWorker(id, address, coordinatorAddress string, gpuCount int, totalRAM int64) *Worker {
	ctx, cancel := context.WithCancel(context.Background())
	return &Worker{
		id:                id,
		address:           address,
		coordinatorAddress: coordinatorAddress,
		rpcServer:         rpc.NewRPCServer(address),
		rpcClient:         rpc.NewRPCClient(coordinatorAddress),
		taskHandlers:      make(map[string]TaskHandler),
		gpuCount:          gpuCount,
		totalRAM:          totalRAM,
		availRAM:          totalRAM,
		status:            cluster.NodeStatusOffline,
		ctx:               ctx,
		cancel:            cancel,
		runningTasks:      make(map[string]context.CancelFunc),
	}
}

// Start starts the worker.
func (w *Worker) Start() error {
	log.Printf("Starting worker %s", w.id)

	// Start RPC server
	if err := w.rpcServer.Start(); err != nil {
		return fmt.Errorf("failed to start RPC server: %v", err)
	}

	// Connect to coordinator
	if err := w.rpcClient.Connect(w.ctx); err != nil {
		return fmt.Errorf("failed to connect to coordinator: %v", err)
	}

	// Register with coordinator
	if err := w.registerWithCoordinator(); err != nil {
		return fmt.Errorf("failed to register with coordinator: %v", err)
	}

	// Start heartbeat
	go w.heartbeat()

	w.status = cluster.NodeStatusOnline
	log.Printf("Worker %s started", w.id)
	return nil
}

// Stop stops the worker.
func (w *Worker) Stop() error {
	log.Printf("Stopping worker %s", w.id)

	// Cancel all running tasks
	w.mu.Lock()
	for taskID, cancelFunc := range w.runningTasks {
		log.Printf("Cancelling task %s", taskID)
		cancelFunc()
	}
	w.mu.Unlock()

	// Cancel context
	w.cancel()

	// Unregister from coordinator
	if err := w.unregisterFromCoordinator(); err != nil {
		log.Printf("Failed to unregister from coordinator: %v", err)
	}

	// Disconnect from coordinator
	if err := w.rpcClient.Disconnect(); err != nil {
		log.Printf("Failed to disconnect from coordinator: %v", err)
	}

	// Stop RPC server
	if err := w.rpcServer.Stop(); err != nil {
		log.Printf("Failed to stop RPC server: %v", err)
	}

	w.status = cluster.NodeStatusOffline
	log.Printf("Worker %s stopped", w.id)
	return nil
}

// RegisterTaskHandler registers a task handler.
func (w *Worker) RegisterTaskHandler(taskType string, handler TaskHandler) error {
	if taskType == "" {
		return errors.New("task type cannot be empty")
	}
	if handler == nil {
		return errors.New("handler cannot be nil")
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	w.taskHandlers[taskType] = handler
	log.Printf("Task handler registered for task type %s", taskType)
	return nil
}

// ExecuteTask executes a task.
func (w *Worker) ExecuteTask(ctx context.Context, taskID, taskType string, taskData map[string]interface{}) (map[string]interface{}, error) {
	w.mu.RLock()
	handler, exists := w.taskHandlers[taskType]
	w.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no handler registered for task type %s", taskType)
	}

	// Update worker status
	w.mu.Lock()
	oldStatus := w.status
	w.status = cluster.NodeStatusBusy
	w.mu.Unlock()

	// Create task context
	taskCtx, cancelFunc := context.WithCancel(ctx)
	w.mu.Lock()
	w.runningTasks[taskID] = cancelFunc
	w.mu.Unlock()

	// Execute task
	log.Printf("Executing task %s of type %s", taskID, taskType)
	result, err := handler(taskCtx, taskData)

	// Clean up
	w.mu.Lock()
	delete(w.runningTasks, taskID)
	w.status = oldStatus
	w.mu.Unlock()

	if err != nil {
		log.Printf("Task %s failed: %v", taskID, err)
		return nil, err
	}

	log.Printf("Task %s completed successfully", taskID)
	return result, nil
}

// CancelTask cancels a task.
func (w *Worker) CancelTask(taskID string) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	cancelFunc, exists := w.runningTasks[taskID]
	if !exists {
		return fmt.Errorf("task %s not found", taskID)
	}

	cancelFunc()
	delete(w.runningTasks, taskID)
	log.Printf("Task %s cancelled", taskID)
	return nil
}

// GetStatus gets the worker status.
func (w *Worker) GetStatus() cluster.NodeStatus {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.status
}

// SetStatus sets the worker status.
func (w *Worker) SetStatus(status cluster.NodeStatus) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.status = status
}

// registerWithCoordinator registers the worker with the coordinator.
func (w *Worker) registerWithCoordinator() error {
	log.Printf("Registering worker %s with coordinator", w.id)
	
	conn, err := w.rpcClient.GetConnection()
	if err != nil {
		return fmt.Errorf("failed to get RPC connection: %v", err)
	}
	
	client := rpc.NewWorkerServiceClient(conn)
	
	req := &rpc.RegisterWorkerRequest{
		WorkerId:  w.id,
		Address:   w.address,
		GpuCount:  int32(w.gpuCount),
		TotalRam:  w.totalRAM,
	}
	
	ctx, cancel := context.WithTimeout(w.ctx, 10*time.Second)
	defer cancel()
	
	resp, err := client.RegisterWorker(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to register worker: %v", err)
	}
	
	if !resp.Success {
		return fmt.Errorf("registration failed: %s", resp.Message)
	}
	
	log.Printf("Worker %s registered successfully", w.id)
	return nil
}

// unregisterFromCoordinator unregisters the worker from the coordinator.
func (w *Worker) unregisterFromCoordinator() error {
	log.Printf("Unregistering worker %s from coordinator", w.id)
	
	conn, err := w.rpcClient.GetConnection()
	if err != nil {
		return fmt.Errorf("failed to get RPC connection: %v", err)
	}
	
	client := rpc.NewWorkerServiceClient(conn)
	
	req := &rpc.UnregisterWorkerRequest{
		WorkerId: w.id,
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	resp, err := client.UnregisterWorker(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to unregister worker: %v", err)
	}
	
	if !resp.Success {
		return fmt.Errorf("unregistration failed: %s", resp.Message)
	}
	
	log.Printf("Worker %s unregistered successfully", w.id)
	return nil
}

// heartbeat sends heartbeat to coordinator.
func (w *Worker) heartbeat() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-w.ctx.Done():
			return
		case <-ticker.C:
			log.Printf("Sending heartbeat from worker %s", w.id)
			
			conn, err := w.rpcClient.GetConnection()
			if err != nil {
				log.Printf("Failed to get RPC connection for heartbeat: %v", err)
				continue
			}
			
			client := rpc.NewWorkerServiceClient(conn)
			
			w.mu.RLock()
			runningTaskIds := make([]string, 0, len(w.runningTasks))
			for taskID := range w.runningTasks {
				runningTaskIds = append(runningTaskIds, taskID)
			}
			status := w.status
			availRAM := w.availRAM
			w.mu.RUnlock()
			
			req := &rpc.HeartbeatRequest{
				WorkerId:       w.id,
				Status:         rpc.NodeStatusToInt32(status),
				RunningTaskIds: runningTaskIds,
				AvailableRam:   availRAM,
			}
			
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			resp, err := client.Heartbeat(ctx, req)
			cancel()
			
			if err != nil {
				log.Printf("Failed to send heartbeat: %v", err)
				continue
			}
			
			if !resp.Success {
				log.Printf("Heartbeat failed: %s", resp.Message)
				continue
			}
			
			log.Printf("Heartbeat sent successfully")
		}
	}
}
