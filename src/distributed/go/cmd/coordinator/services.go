package main

import (
	"context"
	"fmt"
	"log"

	"github.com/MilesONerd/neurenix/src/distributed/go/pkg/cluster"
	"github.com/MilesONerd/neurenix/src/distributed/go/pkg/rpc"
	"github.com/MilesONerd/neurenix/src/distributed/go/pkg/scheduler"
)

type CoordinatorServiceImpl struct {
	cluster   *cluster.Cluster
	scheduler *scheduler.Scheduler
}

func (s *CoordinatorServiceImpl) AssignTask(ctx context.Context, req *rpc.AssignTaskRequest) (*rpc.AssignTaskResponse, error) {
	log.Printf("Assigning task %s of type %s to worker %s", req.TaskId, req.TaskType, req.WorkerId)
	
	taskData := make(map[string]interface{})
	for k, v := range req.TaskData {
		taskData[k] = v
	}
	
	task := &scheduler.Task{
		ID:       req.TaskId,
		Type:     req.TaskType,
		WorkerID: req.WorkerId,
		Data:     taskData,
	}
	
	if err := s.scheduler.SubmitTask(task); err != nil {
		return &rpc.AssignTaskResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to submit task: %v", err),
		}, nil
	}
	
	return &rpc.AssignTaskResponse{
		Success: true,
		Message: "Task assigned successfully",
	}, nil
}

func (s *CoordinatorServiceImpl) GetWorkerStatus(ctx context.Context, req *rpc.GetWorkerStatusRequest) (*rpc.GetWorkerStatusResponse, error) {
	log.Printf("Getting status of worker %s", req.WorkerId)
	
	worker, err := s.cluster.GetWorker(req.WorkerId)
	if err != nil {
		return nil, fmt.Errorf("failed to get worker: %v", err)
	}
	
	status := worker.Status
	
	runningTasks := worker.RunningTasks
	runningTaskIds := make([]string, 0, len(runningTasks))
	for taskID := range runningTasks {
		runningTaskIds = append(runningTaskIds, taskID)
	}
	
	return &rpc.GetWorkerStatusResponse{
		WorkerId:       req.WorkerId,
		Status:         rpc.NodeStatusToInt32(status),
		RunningTaskIds: runningTaskIds,
		AvailableRam:   worker.AvailableRAM,
	}, nil
}

type WorkerServiceImpl struct {
	cluster *cluster.Cluster
}

func (s *WorkerServiceImpl) RegisterWorker(ctx context.Context, req *rpc.RegisterWorkerRequest) (*rpc.RegisterWorkerResponse, error) {
	log.Printf("Registering worker %s at %s", req.WorkerId, req.Address)
	
	worker := &cluster.WorkerNode{
		ID:           req.WorkerId,
		Address:      req.Address,
		GPUCount:     int(req.GpuCount),
		TotalRAM:     req.TotalRam,
		AvailableRAM: req.TotalRam,
		Status:       cluster.NodeStatusOnline,
		RunningTasks: make(map[string]bool),
	}
	
	if err := s.cluster.AddWorker(worker); err != nil {
		return &rpc.RegisterWorkerResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to add worker: %v", err),
		}, nil
	}
	
	log.Printf("Worker %s registered successfully", req.WorkerId)
	return &rpc.RegisterWorkerResponse{
		Success: true,
		Message: "Worker registered successfully",
	}, nil
}

func (s *WorkerServiceImpl) UnregisterWorker(ctx context.Context, req *rpc.UnregisterWorkerRequest) (*rpc.UnregisterWorkerResponse, error) {
	log.Printf("Unregistering worker %s", req.WorkerId)
	
	if err := s.cluster.RemoveWorker(req.WorkerId); err != nil {
		return &rpc.UnregisterWorkerResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to remove worker: %v", err),
		}, nil
	}
	
	log.Printf("Worker %s unregistered successfully", req.WorkerId)
	return &rpc.UnregisterWorkerResponse{
		Success: true,
		Message: "Worker unregistered successfully",
	}, nil
}

func (s *WorkerServiceImpl) Heartbeat(ctx context.Context, req *rpc.HeartbeatRequest) (*rpc.HeartbeatResponse, error) {
	log.Printf("Received heartbeat from worker %s", req.WorkerId)
	
	worker, err := s.cluster.GetWorker(req.WorkerId)
	if err != nil {
		return &rpc.HeartbeatResponse{
			Success: false,
			Message: fmt.Sprintf("Worker not found: %v", err),
		}, nil
	}
	
	worker.Status = rpc.Int32ToNodeStatus(req.Status)
	worker.AvailableRAM = req.AvailableRam
	
	worker.RunningTasks = make(map[string]bool)
	for _, taskID := range req.RunningTaskIds {
		worker.RunningTasks[taskID] = true
	}
	
	if err := s.cluster.UpdateWorker(worker); err != nil {
		return &rpc.HeartbeatResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to update worker: %v", err),
		}, nil
	}
	
	log.Printf("Heartbeat from worker %s processed successfully", req.WorkerId)
	return &rpc.HeartbeatResponse{
		Success: true,
		Message: "Heartbeat processed successfully",
	}, nil
}
