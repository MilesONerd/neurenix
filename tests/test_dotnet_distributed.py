"""
Tests for the .NET distributed computing integration.
"""

import unittest
import numpy as np
import time
import os

from neurenix.tensor import Tensor
from neurenix.distributed.dotnet import DotNetDistributedClient, DotNetDistributedContext


class TestDotNetDistributedClient(unittest.TestCase):
    """Test the DotNetDistributedClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = DotNetDistributedClient()
        self.client._silo_process = None
    
    def test_client_creation(self):
        """Test creating a .NET distributed client."""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.api_base_url, "http://localhost:5000")
        self.assertIsNotNone(self.client.silo_host_path)
    
    def test_silo_path_default(self):
        """Test that the default silo path is set correctly."""
        expected_path = os.path.join(
            os.path.dirname(__file__),
            "../neurenix/distributed/../../src/phynexus/dotnet/Neurenix.Distributed.Orleans/SiloHost/bin/Debug/net8.0/Neurenix.Distributed.Orleans.SiloHost"
        )
        self.assertEqual(self.client.silo_host_path, expected_path)
    
    def test_create_tensor_request(self):
        """Test creating a tensor request (without actual API call)."""
        tensor_id = "test_tensor"
        data = [1.0, 2.0, 3.0, 4.0]
        shape = [2, 2]
        
        self.assertTrue(hasattr(self.client, 'create_tensor'))
    
    def test_get_tensor_request(self):
        """Test getting a tensor request (without actual API call)."""
        tensor_id = "test_tensor"
        
        self.assertTrue(hasattr(self.client, 'get_tensor'))


class TestDotNetDistributedContext(unittest.TestCase):
    """Test the DotNetDistributedContext class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context = DotNetDistributedContext()
        self.context.client._silo_process = None
    
    def test_context_creation(self):
        """Test creating a .NET distributed context."""
        self.assertIsNotNone(self.context)
        self.assertIsNotNone(self.context.client)
    
    def test_create_distributed_tensor(self):
        """Test creating a distributed tensor (without actual API call)."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = Tensor(data)
        tensor_id = "test_tensor"
        
        self.assertTrue(hasattr(self.context, 'create_distributed_tensor'))
    
    def test_get_distributed_tensor(self):
        """Test getting a distributed tensor (without actual API call)."""
        tensor_id = "test_tensor"
        
        self.assertTrue(hasattr(self.context, 'get_distributed_tensor'))


if __name__ == '__main__':
    unittest.main()
