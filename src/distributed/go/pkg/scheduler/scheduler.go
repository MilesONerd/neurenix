// Package scheduler provides task scheduling functionality for distributed computing in Neurenix.
package scheduler

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/MilesONerd/neurenix/src/distributed/go/pkg/cluster"
)

// TaskStatus represents the status of a task.
type TaskStatus int

const (
	// TaskStatusPending indicates that the task is pending execution.
	TaskStatusPending TaskStatus = iota
	// TaskStatusRunning indicates that the task is currently running.
	TaskStatusRunning
	// TaskStatusCompleted indicates that the task has completed successfully.
	TaskStatusCompleted
	// TaskStatusFailed indicates that the task has failed.
	TaskStatusFailed
	// TaskStatusCancelled indicates that the task has been cancelled.
	TaskStatusCancelled
)

// TaskType represents the type of a task.
type TaskType int

const (
	// TaskTypeTraining indicates a model training task.
	TaskTypeTraining TaskType = iota
	// TaskTypeInference indicates a model inference task.
	TaskTypeInference
	// TaskTypeDataProcessing indicates a data processing task.
	TaskTypeDataProcessing
	// TaskTypeCustom indicates a custom task.
	TaskTypeCustom
)

// Task represents a distributed computing task.
type Task struct {
	ID          string
	Type        TaskType
	Status      TaskStatus
	Priority    int
	Dependencies []string
	AssignedNode string
	SubmitTime   time.Time
	StartTime    time.Time
	EndTime      time.Time
	Timeout      time.Duration
	Retries      int
	MaxRetries   int
	Data         map[string]interface{}
	Result       map[string]interface{}
	Error        string
}

// Scheduler represents a task scheduler for distributed computing.
type Scheduler struct {
	mu             sync.RWMutex
	tasks          map[string]*Task
	pendingTasks   []*Task
	runningTasks   map[string]*Task
	completedTasks map[string]*Task
	failedTasks    map[string]*Task
	cluster        *cluster.Cluster
	workerPool     map[string]bool
	ctx            context.Context
	cancel         context.CancelFunc
}

// NewScheduler creates a new scheduler.
func NewScheduler(c *cluster.Cluster) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())
	return &Scheduler{
		tasks:          make(map[string]*Task),
		pendingTasks:   make([]*Task, 0),
		runningTasks:   make(map[string]*Task),
		completedTasks: make(map[string]*Task),
		failedTasks:    make(map[string]*Task),
		cluster:        c,
		workerPool:     make(map[string]bool),
		ctx:            ctx,
		cancel:         cancel,
	}
}

// Start starts the scheduler.
func (s *Scheduler) Start() error {
	log.Println("Starting scheduler")
	go s.scheduleLoop()
	return nil
}

// Stop stops the scheduler.
func (s *Scheduler) Stop() error {
	log.Println("Stopping scheduler")
	s.cancel()
	return nil
}

// SubmitTask submits a task to the scheduler.
func (s *Scheduler) SubmitTask(task *Task) error {
	if task == nil {
		return errors.New("task cannot be nil")
	}
	if task.ID == "" {
		return errors.New("task ID cannot be empty")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.tasks[task.ID]; exists {
		return fmt.Errorf("task %s already exists", task.ID)
	}

	task.Status = TaskStatusPending
	task.SubmitTime = time.Now()
	s.tasks[task.ID] = task
	s.pendingTasks = append(s.pendingTasks, task)

	log.Printf("Task %s submitted", task.ID)
	return nil
}

// CancelTask cancels a task.
func (s *Scheduler) CancelTask(taskID string) error {
	if taskID == "" {
		return errors.New("task ID cannot be empty")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	task, exists := s.tasks[taskID]
	if !exists {
		return fmt.Errorf("task %s not found", taskID)
	}

	if task.Status == TaskStatusCompleted || task.Status == TaskStatusFailed || task.Status == TaskStatusCancelled {
		return fmt.Errorf("task %s already finished with status %v", taskID, task.Status)
	}

	task.Status = TaskStatusCancelled
	if task.Status == TaskStatusRunning {
		delete(s.runningTasks, taskID)
	} else if task.Status == TaskStatusPending {
		for i, t := range s.pendingTasks {
			if t.ID == taskID {
				s.pendingTasks = append(s.pendingTasks[:i], s.pendingTasks[i+1:]...)
				break
			}
		}
	}

	log.Printf("Task %s cancelled", taskID)
	return nil
}

// GetTask gets a task.
func (s *Scheduler) GetTask(taskID string) (*Task, error) {
	if taskID == "" {
		return nil, errors.New("task ID cannot be empty")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	task, exists := s.tasks[taskID]
	if !exists {
		return nil, fmt.Errorf("task %s not found", taskID)
	}

	return task, nil
}

// GetAllTasks gets all tasks.
func (s *Scheduler) GetAllTasks() []*Task {
	s.mu.RLock()
	defer s.mu.RUnlock()

	tasks := make([]*Task, 0, len(s.tasks))
	for _, task := range s.tasks {
		tasks = append(tasks, task)
	}

	return tasks
}

// GetTasksByStatus gets all tasks with a specific status.
func (s *Scheduler) GetTasksByStatus(status TaskStatus) []*Task {
	s.mu.RLock()
	defer s.mu.RUnlock()

	tasks := make([]*Task, 0)
	for _, task := range s.tasks {
		if task.Status == status {
			tasks = append(tasks, task)
		}
	}

	return tasks
}

// UpdateTaskStatus updates the status of a task.
func (s *Scheduler) UpdateTaskStatus(taskID string, status TaskStatus) error {
	if taskID == "" {
		return errors.New("task ID cannot be empty")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	task, exists := s.tasks[taskID]
	if !exists {
		return fmt.Errorf("task %s not found", taskID)
	}

	oldStatus := task.Status
	task.Status = status

	// Update task collections based on status change
	switch status {
	case TaskStatusRunning:
		if oldStatus == TaskStatusPending {
			for i, t := range s.pendingTasks {
				if t.ID == taskID {
					s.pendingTasks = append(s.pendingTasks[:i], s.pendingTasks[i+1:]...)
					break
				}
			}
			task.StartTime = time.Now()
			s.runningTasks[taskID] = task
		}
	case TaskStatusCompleted:
		if oldStatus == TaskStatusRunning {
			delete(s.runningTasks, taskID)
			task.EndTime = time.Now()
			s.completedTasks[taskID] = task
		}
	case TaskStatusFailed:
		if oldStatus == TaskStatusRunning {
			delete(s.runningTasks, taskID)
			task.EndTime = time.Now()
			if task.Retries < task.MaxRetries {
				task.Retries++
				task.Status = TaskStatusPending
				s.pendingTasks = append(s.pendingTasks, task)
				log.Printf("Task %s failed, retrying (%d/%d)", taskID, task.Retries, task.MaxRetries)
			} else {
				s.failedTasks[taskID] = task
				log.Printf("Task %s failed permanently after %d retries", taskID, task.Retries)
			}
		}
	}

	log.Printf("Task %s status updated from %v to %v", taskID, oldStatus, status)
	return nil
}

// scheduleLoop is the main scheduling loop.
func (s *Scheduler) scheduleLoop() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.scheduleTasks()
		}
	}
}

// scheduleTasks schedules pending tasks to available workers.
func (s *Scheduler) scheduleTasks() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.pendingTasks) == 0 {
		return
	}

	// Get available workers
	availableWorkers := s.cluster.GetAvailableWorkers()
	if len(availableWorkers) == 0 {
		return
	}

	// Sort pending tasks by priority (higher priority first)
	// This is a simple implementation; a more sophisticated scheduler would consider
	// task dependencies, worker capabilities, etc.
	for i := 0; i < len(s.pendingTasks); i++ {
		for j := i + 1; j < len(s.pendingTasks); j++ {
			if s.pendingTasks[i].Priority < s.pendingTasks[j].Priority {
				s.pendingTasks[i], s.pendingTasks[j] = s.pendingTasks[j], s.pendingTasks[i]
			}
		}
	}

	// Assign tasks to workers
	workerIndex := 0
	for i := 0; i < len(s.pendingTasks); i++ {
		if workerIndex >= len(availableWorkers) {
			break
		}

		task := s.pendingTasks[i]
		worker := availableWorkers[workerIndex]

		// Check if all dependencies are satisfied
		dependenciesSatisfied := true
		for _, depID := range task.Dependencies {
			depTask, exists := s.tasks[depID]
			if !exists || depTask.Status != TaskStatusCompleted {
				dependenciesSatisfied = false
				break
			}
		}

		if !dependenciesSatisfied {
			continue
		}

		// Assign task to worker
		task.AssignedNode = worker.ID
		task.Status = TaskStatusRunning
		task.StartTime = time.Now()

		// Move task from pending to running
		s.pendingTasks = append(s.pendingTasks[:i], s.pendingTasks[i+1:]...)
		s.runningTasks[task.ID] = task
		i-- // Adjust index since we removed an element

		// Mark worker as busy
		s.cluster.UpdateNodeStatus(worker.ID, cluster.NodeStatusBusy)

		log.Printf("Task %s assigned to worker %s", task.ID, worker.ID)
		workerIndex++
	}
}
