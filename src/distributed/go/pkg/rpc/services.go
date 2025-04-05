package rpc

import (
	"context"
	"fmt"
	"log"

	"github.com/MilesONerd/neurenix/src/distributed/go/pkg/cluster"
	"google.golang.org/grpc"
)

type WorkerServiceServer interface {
	RegisterWorker(context.Context, *RegisterWorkerRequest) (*RegisterWorkerResponse, error)
	UnregisterWorker(context.Context, *UnregisterWorkerRequest) (*UnregisterWorkerResponse, error)
	Heartbeat(context.Context, *HeartbeatRequest) (*HeartbeatResponse, error)
}

type CoordinatorServiceServer interface {
	AssignTask(context.Context, *AssignTaskRequest) (*AssignTaskResponse, error)
	GetWorkerStatus(context.Context, *GetWorkerStatusRequest) (*GetWorkerStatusResponse, error)
}

type WorkerServiceClient interface {
	RegisterWorker(ctx context.Context, in *RegisterWorkerRequest, opts ...grpc.CallOption) (*RegisterWorkerResponse, error)
	UnregisterWorker(ctx context.Context, in *UnregisterWorkerRequest, opts ...grpc.CallOption) (*UnregisterWorkerResponse, error)
	Heartbeat(ctx context.Context, in *HeartbeatRequest, opts ...grpc.CallOption) (*HeartbeatResponse, error)
}

type CoordinatorServiceClient interface {
	AssignTask(ctx context.Context, in *AssignTaskRequest, opts ...grpc.CallOption) (*AssignTaskResponse, error)
	GetWorkerStatus(ctx context.Context, in *GetWorkerStatusRequest, opts ...grpc.CallOption) (*GetWorkerStatusResponse, error)
}

func RegisterWorkerServiceServer(s *grpc.Server, srv WorkerServiceServer) {
	s.RegisterService(&_WorkerService_serviceDesc, srv)
}

func RegisterCoordinatorServiceServer(s *grpc.Server, srv CoordinatorServiceServer) {
	s.RegisterService(&_CoordinatorService_serviceDesc, srv)
}

func NewWorkerServiceClient(cc *grpc.ClientConn) WorkerServiceClient {
	return &workerServiceClient{cc}
}

func NewCoordinatorServiceClient(cc *grpc.ClientConn) CoordinatorServiceClient {
	return &coordinatorServiceClient{cc}
}

type workerServiceClient struct {
	cc *grpc.ClientConn
}

func (c *workerServiceClient) RegisterWorker(ctx context.Context, in *RegisterWorkerRequest, opts ...grpc.CallOption) (*RegisterWorkerResponse, error) {
	out := new(RegisterWorkerResponse)
	err := c.cc.Invoke(ctx, "/neurenix.distributed.WorkerService/RegisterWorker", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerServiceClient) UnregisterWorker(ctx context.Context, in *UnregisterWorkerRequest, opts ...grpc.CallOption) (*UnregisterWorkerResponse, error) {
	out := new(UnregisterWorkerResponse)
	err := c.cc.Invoke(ctx, "/neurenix.distributed.WorkerService/UnregisterWorker", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerServiceClient) Heartbeat(ctx context.Context, in *HeartbeatRequest, opts ...grpc.CallOption) (*HeartbeatResponse, error) {
	out := new(HeartbeatResponse)
	err := c.cc.Invoke(ctx, "/neurenix.distributed.WorkerService/Heartbeat", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

type coordinatorServiceClient struct {
	cc *grpc.ClientConn
}

func (c *coordinatorServiceClient) AssignTask(ctx context.Context, in *AssignTaskRequest, opts ...grpc.CallOption) (*AssignTaskResponse, error) {
	out := new(AssignTaskResponse)
	err := c.cc.Invoke(ctx, "/neurenix.distributed.CoordinatorService/AssignTask", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *coordinatorServiceClient) GetWorkerStatus(ctx context.Context, in *GetWorkerStatusRequest, opts ...grpc.CallOption) (*GetWorkerStatusResponse, error) {
	out := new(GetWorkerStatusResponse)
	err := c.cc.Invoke(ctx, "/neurenix.distributed.CoordinatorService/GetWorkerStatus", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

var _WorkerService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "neurenix.distributed.WorkerService",
	HandlerType: (*WorkerServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "RegisterWorker",
			Handler:    _WorkerService_RegisterWorker_Handler,
		},
		{
			MethodName: "UnregisterWorker",
			Handler:    _WorkerService_UnregisterWorker_Handler,
		},
		{
			MethodName: "Heartbeat",
			Handler:    _WorkerService_Heartbeat_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "services.proto",
}

var _CoordinatorService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "neurenix.distributed.CoordinatorService",
	HandlerType: (*CoordinatorServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "AssignTask",
			Handler:    _CoordinatorService_AssignTask_Handler,
		},
		{
			MethodName: "GetWorkerStatus",
			Handler:    _CoordinatorService_GetWorkerStatus_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "services.proto",
}

func _WorkerService_RegisterWorker_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RegisterWorkerRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerServiceServer).RegisterWorker(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/neurenix.distributed.WorkerService/RegisterWorker",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerServiceServer).RegisterWorker(ctx, req.(*RegisterWorkerRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerService_UnregisterWorker_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(UnregisterWorkerRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerServiceServer).UnregisterWorker(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/neurenix.distributed.WorkerService/UnregisterWorker",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerServiceServer).UnregisterWorker(ctx, req.(*UnregisterWorkerRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerService_Heartbeat_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(HeartbeatRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerServiceServer).Heartbeat(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/neurenix.distributed.WorkerService/Heartbeat",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerServiceServer).Heartbeat(ctx, req.(*HeartbeatRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _CoordinatorService_AssignTask_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(AssignTaskRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(CoordinatorServiceServer).AssignTask(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/neurenix.distributed.CoordinatorService/AssignTask",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(CoordinatorServiceServer).AssignTask(ctx, req.(*AssignTaskRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _CoordinatorService_GetWorkerStatus_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetWorkerStatusRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(CoordinatorServiceServer).GetWorkerStatus(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/neurenix.distributed.CoordinatorService/GetWorkerStatus",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(CoordinatorServiceServer).GetWorkerStatus(ctx, req.(*GetWorkerStatusRequest))
	}
	return interceptor(ctx, in, info, handler)
}

type RegisterWorkerRequest struct {
	WorkerId  string
	Address   string
	GpuCount  int32
	TotalRam  int64
}

type RegisterWorkerResponse struct {
	Success bool
	Message string
}

type UnregisterWorkerRequest struct {
	WorkerId string
}

type UnregisterWorkerResponse struct {
	Success bool
	Message string
}

type HeartbeatRequest struct {
	WorkerId       string
	Status         int32
	RunningTaskIds []string
	AvailableRam   int64
}

type HeartbeatResponse struct {
	Success bool
	Message string
}

type AssignTaskRequest struct {
	TaskId    string
	TaskType  string
	WorkerId  string
	TaskData  map[string]string
}

type AssignTaskResponse struct {
	Success bool
	Message string
	Result  map[string]string
}

type GetWorkerStatusRequest struct {
	WorkerId string
}

type GetWorkerStatusResponse struct {
	WorkerId       string
	Status         int32
	RunningTaskIds []string
	AvailableRam   int64
}

func NodeStatusToInt32(status cluster.NodeStatus) int32 {
	return int32(status)
}

func Int32ToNodeStatus(status int32) cluster.NodeStatus {
	return cluster.NodeStatus(status)
}
