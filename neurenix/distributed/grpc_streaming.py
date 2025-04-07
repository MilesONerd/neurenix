"""
gRPC-Streaming support for Neurenix distributed training.

This module provides streaming capabilities for distributed training and inference
using gRPC streaming APIs. It supports server-side streaming, client-side streaming,
and bidirectional streaming for efficient data transfer and real-time communication.
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Union, Iterator, Tuple

import grpc
from concurrent import futures

from neurenix.distributed.base import DistributedBackend
from neurenix.tensor import Tensor
from neurenix.device import get_device

try:
    from neurenix.distributed.proto import services_pb2
    from neurenix.distributed.proto import services_pb2_grpc
except ImportError:
    logging.warning("gRPC protobuf modules not found. Streaming functionality will be limited.")
    services_pb2 = None
    services_pb2_grpc = None

class StreamingClient:
    """Client for gRPC streaming operations."""
    
    def __init__(self, address: str, secure: bool = False, credentials: Optional[grpc.ChannelCredentials] = None):
        """Initialize a streaming client."""
        self.address = address
        self.secure = secure
        self.credentials = credentials
        self.channel = None
        self.worker_stub = None
        self.coordinator_stub = None
        self.data_stub = None
        self._connected = False
        self._lock = threading.RLock()
        self.active_streams = {}
        self.stream_callbacks = {}
        
    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to the gRPC server."""
        with self._lock:
            if self._connected:
                return True
            
            try:
                if services_pb2 is None or services_pb2_grpc is None:
                    logging.error("Cannot connect: gRPC protobuf modules not found")
                    return False
                
                if self.secure and self.credentials:
                    self.channel = grpc.secure_channel(self.address, self.credentials)
                else:
                    self.channel = grpc.insecure_channel(self.address)
                
                self.worker_stub = services_pb2_grpc.WorkerServiceStub(self.channel)
                self.coordinator_stub = services_pb2_grpc.CoordinatorServiceStub(self.channel)
                self.data_stub = services_pb2_grpc.DataStreamingServiceStub(self.channel)
                
                grpc.channel_ready_future(self.channel).result(timeout=timeout)
                self._connected = True
                logging.info(f"Connected to gRPC server at {self.address}")
                return True
                
            except Exception as e:
                logging.error(f"Error connecting to gRPC server: {str(e)}")
                self._cleanup()
                return False
    
    def disconnect(self) -> None:
        """Disconnect from the gRPC server and clean up resources."""
        with self._lock:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        for stream_id, stream_info in list(self.active_streams.items()):
            try:
                if 'stream' in stream_info and hasattr(stream_info['stream'], 'cancel'):
                    stream_info['stream'].cancel()
                if 'thread' in stream_info and stream_info['thread'].is_alive():
                    stream_info['thread'].join(timeout=1.0)
            except Exception as e:
                logging.warning(f"Error cleaning up stream {stream_id}: {str(e)}")
        
        self.active_streams.clear()
        self.stream_callbacks.clear()
        
        if self.channel is not None:
            try:
                self.channel.close()
            except Exception as e:
                logging.warning(f"Error closing gRPC channel: {str(e)}")
            
            self.channel = None
            self.worker_stub = None
            self.coordinator_stub = None
            self.data_stub = None
            
        self._connected = False
    
    def stream_logs(self, worker_id: str, log_level: str, component: str, 
                   messages: Iterator[str]) -> Tuple[bool, int]:
        """Stream logs to the coordinator."""
        if not self._ensure_connected():
            return False, 0
        
        try:
            count = 0
            log_stream = self.worker_stub.StreamLogs()
            
            for message in messages:
                log_msg = services_pb2.LogMessage(
                    worker_id=worker_id,
                    log_level=log_level,
                    message=message,
                    timestamp=int(time.time() * 1000),
                    component=component
                )
                log_stream.write(log_msg)
                count += 1
            
            response = log_stream.done_writing()
            return response.success, count
            
        except Exception as e:
            logging.error(f"Error streaming logs: {str(e)}")
            return False, count
    
    def stream_metrics(self, worker_id: str, metrics: Iterator[Dict[str, Any]]) -> Tuple[bool, int]:
        """Stream metrics to the coordinator."""
        if not self._ensure_connected():
            return False, 0
        
        try:
            count = 0
            metric_stream = self.worker_stub.StreamMetrics()
            
            for metric in metrics:
                metric_msg = services_pb2.MetricMessage(
                    worker_id=worker_id,
                    metric_name=metric.get('name', ''),
                    value=float(metric.get('value', 0.0)),
                    timestamp=int(metric.get('timestamp', time.time() * 1000)),
                    labels=metric.get('labels', {})
                )
                metric_stream.write(metric_msg)
                count += 1
            
            response = metric_stream.done_writing()
            return response.success, count
            
        except Exception as e:
            logging.error(f"Error streaming metrics: {str(e)}")
            return False, count
    
    def subscribe_task_updates(self, worker_id: str, task_id: str, 
                              callback: Callable[[Dict[str, Any]], None]) -> str:
        """Subscribe to task updates from the coordinator."""
        if not self._ensure_connected():
            return ""
        
        stream_id = f"task_{worker_id}_{task_id}_{time.time()}"
        
        def task_update_handler():
            try:
                request = services_pb2.TaskUpdateRequest(
                    worker_id=worker_id,
                    task_id=task_id
                )
                
                for update in self.coordinator_stub.StreamTaskUpdates(request):
                    update_data = {
                        'task_id': update.task_id,
                        'update_type': update.update_type,
                        'update_data': dict(update.update_data),
                        'timestamp': update.timestamp
                    }
                    callback(update_data)
                    
            except Exception as e:
                logging.error(f"Error in task update handler: {str(e)}")
            finally:
                with self._lock:
                    if stream_id in self.active_streams:
                        del self.active_streams[stream_id]
        
        handler_thread = threading.Thread(target=task_update_handler, daemon=True)
        handler_thread.start()
        
        with self._lock:
            self.active_streams[stream_id] = {
                'type': 'task_updates',
                'worker_id': worker_id,
                'task_id': task_id,
                'thread': handler_thread
            }
        
        return stream_id
    
    def bidirectional_stream(self, sender_id: str, 
                            message_callback: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> str:
        """Start a bidirectional stream with the coordinator."""
        if not self._ensure_connected():
            return ""
        
        stream_id = f"bidirectional_{sender_id}_{time.time()}"
        message_queue = queue.Queue()
        
        def bidirectional_handler():
            try:
                stream = self.coordinator_stub.BidirectionalStream()
                
                with self._lock:
                    if stream_id in self.active_streams:
                        self.active_streams[stream_id]['stream'] = stream
                
                def send_messages():
                    seq_num = 0
                    try:
                        while True:
                            try:
                                payload, msg_type = message_queue.get(timeout=0.1)
                                
                                request = services_pb2.StreamRequest(
                                    sender_id=sender_id,
                                    message_type=msg_type,
                                    payload=payload,
                                    sequence_number=seq_num
                                )
                                
                                stream.write(request)
                                seq_num += 1
                                
                            except queue.Empty:
                                with self._lock:
                                    if stream_id not in self.active_streams:
                                        break
                    except Exception as e:
                        logging.error(f"Error in send_messages: {str(e)}")
                
                sender_thread = threading.Thread(target=send_messages, daemon=True)
                sender_thread.start()
                
                for response in stream:
                    response_data = {
                        'sender_id': response.sender_id,
                        'message_type': response.message_type,
                        'payload': response.payload,
                        'sequence_number': response.sequence_number,
                        'requires_ack': response.requires_ack
                    }
                    
                    reply = message_callback(response_data)
                    
                    if reply is not None and isinstance(reply, dict):
                        payload = reply.get('payload', b'')
                        msg_type = reply.get('message_type', 'response')
                        message_queue.put((payload, msg_type))
                
            except Exception as e:
                logging.error(f"Error in bidirectional handler: {str(e)}")
            finally:
                with self._lock:
                    if stream_id in self.active_streams:
                        del self.active_streams[stream_id]
        
        handler_thread = threading.Thread(target=bidirectional_handler, daemon=True)
        handler_thread.start()
        
        with self._lock:
            self.active_streams[stream_id] = {
                'type': 'bidirectional',
                'sender_id': sender_id,
                'thread': handler_thread,
                'queue': message_queue
            }
        
        return stream_id
    
    def distribute_data(self, dataset_id: str, batch_size: int = 32, 
                       shuffle: bool = False, options: Dict[str, str] = None) -> Iterator[Dict[str, Any]]:
        """Receive distributed data from the server."""
        if not self._ensure_connected():
            return
        
        try:
            request = services_pb2.DataRequest(
                dataset_id=dataset_id,
                batch_size=batch_size,
                shuffle=shuffle,
                options=options or {}
            )
            
            for chunk in self.data_stub.DistributeData(request):
                yield {
                    'dataset_id': chunk.dataset_id,
                    'chunk_index': chunk.chunk_index,
                    'data': chunk.data,
                    'format': chunk.format,
                    'metadata': dict(chunk.metadata)
                }
                
        except Exception as e:
            logging.error(f"Error in distribute_data: {str(e)}")
    
    def process_data_stream(self, process_id: str, 
                           requests: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Process data using bidirectional streaming."""
        if not self._ensure_connected():
            return
        
        try:
            process_stream = self.data_stub.ProcessDataStream()
            
            def send_requests():
                try:
                    for req in requests:
                        request = services_pb2.ProcessRequest(
                            process_id=process_id,
                            operation=req.get('operation', ''),
                            input_data=req.get('input_data', b''),
                            parameters=req.get('parameters', {})
                        )
                        process_stream.write(request)
                    
                    process_stream.done_writing()
                except Exception as e:
                    logging.error(f"Error sending process requests: {str(e)}")
            
            sender_thread = threading.Thread(target=send_requests, daemon=True)
            sender_thread.start()
            
            for response in process_stream:
                yield {
                    'process_id': response.process_id,
                    'success': response.success,
                    'output_data': response.output_data,
                    'message': response.message,
                    'metadata': dict(response.metadata)
                }
                
            sender_thread.join()
            
        except Exception as e:
            logging.error(f"Error in process_data_stream: {str(e)}")
    
    def _ensure_connected(self) -> bool:
        """Ensure the client is connected to the server."""
        with self._lock:
            if not self._connected:
                return self.connect()
            return True


class StreamingServer:
    """Server for gRPC streaming operations."""
    
    def __init__(self, address: str, secure: bool = False, 
                server_credentials: Optional[grpc.ServerCredentials] = None,
                max_workers: int = 10):
        """Initialize a streaming server."""
        self.address = address
        self.secure = secure
        self.credentials = server_credentials
        self.max_workers = max_workers
        self.server = None
        self.servicer = None
        self._running = False
        self._lock = threading.RLock()
        
        self.log_handlers = []
        self.metric_handlers = []
        self.task_update_publishers = {}
        self.data_chunk_publishers = {}
        self.process_handlers = {}
        self.bidirectional_handlers = []
    
    def start(self, blocking: bool = False) -> bool:
        """Start the gRPC server."""
        with self._lock:
            if self._running:
                return True
            
            try:
                if services_pb2 is None or services_pb2_grpc is None:
                    logging.error("Cannot start server: gRPC protobuf modules not found")
                    return False
                
                self.server = grpc.server(
                    futures.ThreadPoolExecutor(max_workers=self.max_workers)
                )
                
                self.servicer = StreamingServicer(self)
                services_pb2_grpc.add_WorkerServiceServicer_to_server(self.servicer, self.server)
                services_pb2_grpc.add_CoordinatorServiceServicer_to_server(self.servicer, self.server)
                services_pb2_grpc.add_DataStreamingServiceServicer_to_server(self.servicer, self.server)
                
                if self.secure and self.credentials:
                    self.server.add_secure_port(self.address, self.credentials)
                else:
                    self.server.add_insecure_port(self.address)
                
                self.server.start()
                self._running = True
                
                logging.info(f"gRPC streaming server started on {self.address}")
                
                if blocking:
                    self.server.wait_for_termination()
                
                return True
                
            except Exception as e:
                logging.error(f"Error starting gRPC server: {str(e)}")
                self._cleanup()
                return False
    
    def stop(self, grace: float = 5.0) -> None:
        """Stop the gRPC server."""
        with self._lock:
            self._cleanup(grace)
    
    def _cleanup(self, grace: float = 5.0) -> None:
        """Clean up resources."""
        if self.server is not None:
            try:
                self.server.stop(grace)
            except Exception as e:
                logging.warning(f"Error stopping gRPC server: {str(e)}")
            
            self.server = None
            self.servicer = None
            
        self._running = False


class StreamingServicer:
    """Implementation of gRPC streaming services."""
    
    def __init__(self, server: StreamingServer):
        """Initialize the servicer."""
        self.server = server
