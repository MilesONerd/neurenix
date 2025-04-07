package rpc

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type StreamingServer struct {
	UnimplementedWorkerServiceServer
	UnimplementedCoordinatorServiceServer
	UnimplementedDataStreamingServiceServer
	
	logStreams     map[string][]chan *LogMessage
	metricStreams  map[string][]chan *MetricMessage
	taskStreams    map[string][]chan *TaskUpdateResponse
	dataStreams    map[string][]chan *DataChunk
	processStreams map[string][]chan *ProcessResponse
	
	bidirectionalStreams map[string]map[string]chan *StreamResponse
	
	logMu     sync.RWMutex
	metricMu  sync.RWMutex
	taskMu    sync.RWMutex
	dataMu    sync.RWMutex
	processMu sync.RWMutex
	biMu      sync.RWMutex
}

func NewStreamingServer() *StreamingServer {
	return &StreamingServer{
		logStreams:           make(map[string][]chan *LogMessage),
		metricStreams:        make(map[string][]chan *MetricMessage),
		taskStreams:          make(map[string][]chan *TaskUpdateResponse),
		dataStreams:          make(map[string][]chan *DataChunk),
		processStreams:       make(map[string][]chan *ProcessResponse),
		bidirectionalStreams: make(map[string]map[string]chan *StreamResponse),
	}
}

func (s *StreamingServer) RegisterWithServer(server *grpc.Server) {
	RegisterWorkerServiceServer(server, s)
	RegisterCoordinatorServiceServer(server, s)
	RegisterDataStreamingServiceServer(server, s)
}

func (s *StreamingServer) StreamLogs(stream WorkerService_StreamLogsServer) error {
	var workerId string
	var count int32
	
	for {
		logMsg, err := stream.Recv()
		if err == io.EOF {
			return stream.SendAndClose(&LogResponse{
				Success:       true,
				ReceivedCount: count,
			})
		}
		if err != nil {
			log.Printf("Error receiving log message: %v", err)
			return err
		}
		
		if workerId == "" {
			workerId = logMsg.WorkerId
			log.Printf("Started log stream for worker %s", workerId)
		}
		
		s.logMu.RLock()
		channels, exists := s.logStreams[workerId]
		s.logMu.RUnlock()
		
		if exists {
			for _, ch := range channels {
				select {
				case ch <- logMsg:
				default:
					log.Printf("Log channel full for worker %s, skipping message", workerId)
				}
			}
		}
		
		count++
	}
}

func (s *StreamingServer) StreamMetrics(stream WorkerService_StreamMetricsServer) error {
	var workerId string
	var count int32
	
	for {
		metricMsg, err := stream.Recv()
		if err == io.EOF {
			return stream.SendAndClose(&MetricResponse{
				Success:       true,
				ReceivedCount: count,
			})
		}
		if err != nil {
			log.Printf("Error receiving metric message: %v", err)
			return err
		}
		
		if workerId == "" {
			workerId = metricMsg.WorkerId
			log.Printf("Started metric stream for worker %s", workerId)
		}
		
		s.metricMu.RLock()
		channels, exists := s.metricStreams[workerId]
		s.metricMu.RUnlock()
		
		if exists {
			for _, ch := range channels {
				select {
				case ch <- metricMsg:
				default:
					log.Printf("Metric channel full for worker %s, skipping message", workerId)
				}
			}
		}
		
		count++
	}
}

func (s *StreamingServer) StreamTaskUpdates(req *TaskUpdateRequest, stream CoordinatorService_StreamTaskUpdatesServer) error {
	workerId := req.WorkerId
	taskId := req.TaskId
	
	log.Printf("Starting task update stream for worker %s, task %s", workerId, taskId)
	
	key := fmt.Sprintf("%s:%s", workerId, taskId)
	
	ch := make(chan *TaskUpdateResponse, 100)
	
	s.taskMu.Lock()
	if _, exists := s.taskStreams[key]; !exists {
		s.taskStreams[key] = []chan *TaskUpdateResponse{}
	}
	s.taskStreams[key] = append(s.taskStreams[key], ch)
	s.taskMu.Unlock()
	
	defer func() {
		s.taskMu.Lock()
		defer s.taskMu.Unlock()
		
		channels := s.taskStreams[key]
		for i, c := range channels {
			if c == ch {
				s.taskStreams[key] = append(channels[:i], channels[i+1:]...)
				break
			}
		}
		
		if len(s.taskStreams[key]) == 0 {
			delete(s.taskStreams, key)
		}
		
		close(ch)
	}()
	
	for {
		select {
		case <-stream.Context().Done():
			return nil
		case update, ok := <-ch:
			if !ok {
				return nil
			}
			
			if err := stream.Send(update); err != nil {
				log.Printf("Error sending task update: %v", err)
				return err
			}
		}
	}
}

func (s *StreamingServer) BidirectionalStream(stream CoordinatorService_BidirectionalStreamServer) error {
	var senderId string
	
	responseCh := make(chan *StreamResponse, 100)
	
	go func() {
		for {
			select {
			case <-stream.Context().Done():
				return
			case resp, ok := <-responseCh:
				if !ok {
					return
				}
				
				if err := stream.Send(resp); err != nil {
					log.Printf("Error sending stream response: %v", err)
					return
				}
			}
		}
	}()
	
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			log.Printf("Error receiving stream request: %v", err)
			return err
		}
		
		if senderId == "" {
			senderId = req.SenderId
			log.Printf("Started bidirectional stream for sender %s", senderId)
			
			s.biMu.Lock()
			if _, exists := s.bidirectionalStreams[senderId]; !exists {
				s.bidirectionalStreams[senderId] = make(map[string]chan *StreamResponse)
			}
			s.bidirectionalStreams[senderId]["self"] = responseCh
			s.biMu.Unlock()
			
			defer func() {
				s.biMu.Lock()
				defer s.biMu.Unlock()
				
				if streams, exists := s.bidirectionalStreams[senderId]; exists {
					delete(streams, "self")
					if len(streams) == 0 {
						delete(s.bidirectionalStreams, senderId)
					}
				}
				
				close(responseCh)
			}()
		}
		
		resp := &StreamResponse{
			SenderId:       "coordinator",
			MessageType:    req.MessageType,
			Payload:        req.Payload,
			SequenceNumber: req.SequenceNumber,
			RequiresAck:    false,
		}
		
		select {
		case responseCh <- resp:
		default:
			log.Printf("Response channel full for sender %s, skipping response", senderId)
		}
	}
}

func (s *StreamingServer) DistributeData(req *DataRequest, stream DataStreamingService_DistributeDataServer) error {
	datasetId := req.DatasetId
	
	log.Printf("Starting data distribution stream for dataset %s", datasetId)
	
	ch := make(chan *DataChunk, 100)
	
	s.dataMu.Lock()
	if _, exists := s.dataStreams[datasetId]; !exists {
		s.dataStreams[datasetId] = []chan *DataChunk{}
	}
	s.dataStreams[datasetId] = append(s.dataStreams[datasetId], ch)
	s.dataMu.Unlock()
	
	defer func() {
		s.dataMu.Lock()
		defer s.dataMu.Unlock()
		
		channels := s.dataStreams[datasetId]
		for i, c := range channels {
			if c == ch {
				s.dataStreams[datasetId] = append(channels[:i], channels[i+1:]...)
				break
			}
		}
		
		if len(s.dataStreams[datasetId]) == 0 {
			delete(s.dataStreams, datasetId)
		}
		
		close(ch)
	}()
	
	for {
		select {
		case <-stream.Context().Done():
			return nil
		case chunk, ok := <-ch:
			if !ok {
				return nil
			}
			
			if err := stream.Send(chunk); err != nil {
				log.Printf("Error sending data chunk: %v", err)
				return err
			}
		}
	}
}

func (s *StreamingServer) CollectData(stream DataStreamingService_CollectDataServer) error {
	var datasetId string
	var count int32
	
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			return stream.SendAndClose(&DataResponse{
				Success:        true,
				ChunksReceived: count,
				DatasetId:      datasetId,
			})
		}
		if err != nil {
			log.Printf("Error receiving data chunk: %v", err)
			return err
		}
		
		if datasetId == "" {
			datasetId = chunk.DatasetId
			log.Printf("Started data collection stream for dataset %s", datasetId)
		}
		
		
		count++
	}
}

func (s *StreamingServer) ProcessDataStream(stream DataStreamingService_ProcessDataStreamServer) error {
	var processId string
	
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			log.Printf("Error receiving process request: %v", err)
			return err
		}
		
		if processId == "" {
			processId = req.ProcessId
			log.Printf("Started data processing stream for process %s", processId)
		}
		
		resp := &ProcessResponse{
			ProcessId:  req.ProcessId,
			Success:    true,
			OutputData: req.InputData,
			Message:    fmt.Sprintf("Processed operation: %s", req.Operation),
			Metadata:   req.Parameters,
		}
		
		if err := stream.Send(resp); err != nil {
			log.Printf("Error sending process response: %v", err)
			return err
		}
	}
}

func (s *StreamingServer) PublishTaskUpdate(workerId, taskId string, updateType string, updateData map[string]string) {
	key := fmt.Sprintf("%s:%s", workerId, taskId)
	
	update := &TaskUpdateResponse{
		TaskId:     taskId,
		UpdateType: updateType,
		UpdateData: updateData,
		Timestamp:  time.Now().UnixNano(),
	}
	
	s.taskMu.RLock()
	channels, exists := s.taskStreams[key]
	s.taskMu.RUnlock()
	
	if exists {
		for _, ch := range channels {
			select {
			case ch <- update:
			default:
				log.Printf("Task update channel full for %s, skipping update", key)
			}
		}
	}
}

func (s *StreamingServer) PublishDataChunk(datasetId string, chunkIndex int32, data []byte, format string, metadata map[string]string) {
	chunk := &DataChunk{
		DatasetId:   datasetId,
		ChunkIndex:  chunkIndex,
		Data:        data,
		Format:      format,
		Metadata:    metadata,
	}
	
	s.dataMu.RLock()
	channels, exists := s.dataStreams[datasetId]
	s.dataMu.RUnlock()
	
	if exists {
		for _, ch := range channels {
			select {
			case ch <- chunk:
			default:
				log.Printf("Data chunk channel full for dataset %s, skipping chunk", datasetId)
			}
		}
	}
}

func (s *StreamingServer) PublishProcessResponse(processId string, success bool, outputData []byte, message string, metadata map[string]string) {
	resp := &ProcessResponse{
		ProcessId:  processId,
		Success:    success,
		OutputData: outputData,
		Message:    message,
		Metadata:   metadata,
	}
	
	s.processMu.RLock()
	channels, exists := s.processStreams[processId]
	s.processMu.RUnlock()
	
	if exists {
		for _, ch := range channels {
			select {
			case ch <- resp:
			default:
				log.Printf("Process response channel full for process %s, skipping response", processId)
			}
		}
	}
}

func (s *StreamingServer) SendBidirectionalMessage(targetId, senderId, messageType string, payload []byte, sequenceNumber int64, requiresAck bool) error {
	s.biMu.RLock()
	senderStreams, exists := s.bidirectionalStreams[targetId]
	s.biMu.RUnlock()
	
	if !exists || len(senderStreams) == 0 {
		return errors.New("target not found or not connected")
	}
	
	resp := &StreamResponse{
		SenderId:       senderId,
		MessageType:    messageType,
		Payload:        payload,
		SequenceNumber: sequenceNumber,
		RequiresAck:    requiresAck,
	}
	
	for _, ch := range senderStreams {
		select {
		case ch <- resp:
		default:
			log.Printf("Bidirectional channel full for sender %s, skipping message", targetId)
		}
	}
	
	return nil
}
