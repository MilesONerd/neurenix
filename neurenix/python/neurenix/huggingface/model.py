"""
Hugging Face model integration for Neurenix.

This module provides functionality for loading, fine-tuning, and using
Hugging Face models in Neurenix.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import transformers
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from neurenix.tensor import Tensor
from neurenix.nn.module import Module


class HuggingFaceModel(Module):
    """
    Hugging Face model wrapper for Neurenix.
    
    This class wraps Hugging Face models for use in Neurenix,
    enabling seamless integration with the rest of the framework.
    """
    
    def __init__(
        self,
        model_name: str,
        task: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        name: str = "HuggingFaceModel",
    ):
        """
        Initialize Hugging Face model.
        
        Args:
            model_name: Name of the model to load
            task: Task for which to load the model
            config: Model configuration
            cache_dir: Directory to cache models
            device: Device to load the model on
            name: Model name
        """
        super().__init__(name=name)
        
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "Hugging Face is not available. Please install it with 'pip install transformers torch'."
            )
        
        self.model_name = model_name
        self.task = task
        self.config = config or {}
        self.cache_dir = cache_dir
        self.device = device
        
        # Load model
        self.model = self._load_model()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _load_model(self):
        """
        Load Hugging Face model.
        
        Returns:
            Hugging Face model
        """
        if self.task is not None:
            # Load model for specific task
            model = transformers.pipeline(
                task=self.task,
                model=self.model_name,
                config=self.config,
                device=self.device,
            )
        else:
            # Load model directly
            model = transformers.AutoModel.from_pretrained(
                self.model_name,
                config=self.config,
                cache_dir=self.cache_dir,
            )
            
            # Move model to device
            model = model.to(self.device)
        
        return model
    
    def forward(self, x: Union[Tensor, np.ndarray, str, List[str]]) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input data
            
        Returns:
            Model output
        """
        # Convert input to appropriate format
        if isinstance(x, Tensor):
            # Convert to torch tensor
            x_torch = torch.tensor(x.numpy())
        elif isinstance(x, np.ndarray):
            # Convert to torch tensor
            x_torch = torch.tensor(x)
        else:
            # Use as is (string or list of strings)
            x_torch = x
        
        # Forward pass
        with torch.no_grad():
            if self.task is not None:
                # Use pipeline
                output = self.model(x_torch)
            else:
                # Use model directly
                output = self.model(x_torch)
        
        # Convert output to Tensor
        if isinstance(output, torch.Tensor):
            # Convert torch tensor to numpy
            output_np = output.cpu().numpy()
            
            # Convert to Tensor
            return Tensor(output_np)
        elif isinstance(output, dict):
            # Convert dict of torch tensors to dict of Tensors
            output_dict = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    output_dict[key] = Tensor(value.cpu().numpy())
                else:
                    output_dict[key] = value
            
            return output_dict
        else:
            # Return as is
            return output
    
    def to(self, device: str) -> "HuggingFaceModel":
        """
        Move model to device.
        
        Args:
            device: Device to move model to
            
        Returns:
            Self
        """
        self.device = device
        
        # Move model to device
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        
        return self
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save model to
        """
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        else:
            # Save model configuration
            config = {
                "model_name": self.model_name,
                "task": self.task,
                "config": self.config,
                "device": self.device,
                "name": self.name,
            }
            
            # Save configuration
            import json
            with open(f"{path}_config.json", "w") as f:
                json.dump(config, f)
    
    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        if hasattr(self.model, "from_pretrained"):
            self.model = self.model.from_pretrained(path)
            
            # Move model to device
            self.model = self.model.to(self.device)
        else:
            # Load configuration
            import json
            with open(f"{path}_config.json", "r") as f:
                config = json.load(f)
            
            # Update attributes
            self.model_name = config["model_name"]
            self.task = config["task"]
            self.config = config["config"]
            self.device = config["device"]
            self.name = config["name"]
            
            # Load model
            self.model = self._load_model()


class HuggingFaceTextModel(HuggingFaceModel):
    """
    Hugging Face text model wrapper for Neurenix.
    
    This class wraps Hugging Face text models for use in Neurenix,
    providing additional functionality for text processing.
    """
    
    def __init__(
        self,
        model_name: str,
        task: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        tokenizer_name: Optional[str] = None,
        max_length: int = 512,
        name: str = "HuggingFaceTextModel",
    ):
        """
        Initialize Hugging Face text model.
        
        Args:
            model_name: Name of the model to load
            task: Task for which to load the model
            config: Model configuration
            cache_dir: Directory to cache models
            device: Device to load the model on
            tokenizer_name: Name of the tokenizer to load
            max_length: Maximum sequence length
            name: Model name
        """
        super().__init__(
            model_name=model_name,
            task=task,
            config=config,
            cache_dir=cache_dir,
            device=device,
            name=name,
        )
        
        self.tokenizer_name = tokenizer_name or model_name
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
    
    def _load_tokenizer(self):
        """
        Load Hugging Face tokenizer.
        
        Returns:
            Hugging Face tokenizer
        """
        return transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.cache_dir,
        )
    
    def forward(self, x: Union[str, List[str]]) -> Dict[str, Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input text
            
        Returns:
            Model output
        """
        # Tokenize input
        inputs = self.tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            if self.task is not None:
                # Use pipeline
                output = self.model(x)
            else:
                # Use model directly
                output = self.model(**inputs)
        
        # Convert output to Tensor
        if isinstance(output, torch.Tensor):
            # Convert torch tensor to numpy
            output_np = output.cpu().numpy()
            
            # Convert to Tensor
            return Tensor(output_np)
        elif hasattr(output, "to_tuple"):
            # Convert tuple of torch tensors to tuple of Tensors
            output_tuple = output.to_tuple()
            return tuple(Tensor(t.cpu().numpy()) for t in output_tuple if isinstance(t, torch.Tensor))
        elif isinstance(output, dict):
            # Convert dict of torch tensors to dict of Tensors
            output_dict = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    output_dict[key] = Tensor(value.cpu().numpy())
                else:
                    output_dict[key] = value
            
            return output_dict
        else:
            # Return as is
            return output
    
    def encode(self, x: Union[str, List[str]]) -> Dict[str, Tensor]:
        """
        Encode text using the tokenizer.
        
        Args:
            x: Input text
            
        Returns:
            Encoded text
        """
        # Tokenize input
        inputs = self.tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Convert to Tensor
        encoded = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                encoded[key] = Tensor(value.numpy())
            else:
                encoded[key] = value
        
        return encoded
    
    def decode(self, x: Union[Tensor, np.ndarray, List[int]]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            x: Token IDs
            
        Returns:
            Decoded text
        """
        # Convert to list of ints
        if isinstance(x, Tensor):
            token_ids = x.numpy().tolist()
        elif isinstance(x, np.ndarray):
            token_ids = x.tolist()
        else:
            token_ids = x
        
        # Decode
        return self.tokenizer.decode(token_ids)
    
    def save(self, path: str):
        """
        Save model and tokenizer to disk.
        
        Args:
            path: Path to save model to
        """
        # Save model
        super().save(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{path}_tokenizer")
    
    def load(self, path: str):
        """
        Load model and tokenizer from disk.
        
        Args:
            path: Path to load model from
        """
        # Load model
        super().load(path)
        
        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f"{path}_tokenizer")


class HuggingFaceVisionModel(HuggingFaceModel):
    """
    Hugging Face vision model wrapper for Neurenix.
    
    This class wraps Hugging Face vision models for use in Neurenix,
    providing additional functionality for image processing.
    """
    
    def __init__(
        self,
        model_name: str,
        task: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        processor_name: Optional[str] = None,
        name: str = "HuggingFaceVisionModel",
    ):
        """
        Initialize Hugging Face vision model.
        
        Args:
            model_name: Name of the model to load
            task: Task for which to load the model
            config: Model configuration
            cache_dir: Directory to cache models
            device: Device to load the model on
            processor_name: Name of the processor to load
            name: Model name
        """
        super().__init__(
            model_name=model_name,
            task=task,
            config=config,
            cache_dir=cache_dir,
            device=device,
            name=name,
        )
        
        self.processor_name = processor_name or model_name
        
        # Load processor
        self.processor = self._load_processor()
    
    def _load_processor(self):
        """
        Load Hugging Face processor.
        
        Returns:
            Hugging Face processor
        """
        return transformers.AutoImageProcessor.from_pretrained(
            self.processor_name,
            cache_dir=self.cache_dir,
        )
    
    def forward(self, x: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input images
            
        Returns:
            Model output
        """
        # Process input
        inputs = self.processor(x, return_tensors="pt")
        
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            if self.task is not None:
                # Use pipeline
                output = self.model(x)
            else:
                # Use model directly
                output = self.model(**inputs)
        
        # Convert output to Tensor
        if isinstance(output, torch.Tensor):
            # Convert torch tensor to numpy
            output_np = output.cpu().numpy()
            
            # Convert to Tensor
            return Tensor(output_np)
        elif hasattr(output, "to_tuple"):
            # Convert tuple of torch tensors to tuple of Tensors
            output_tuple = output.to_tuple()
            return tuple(Tensor(t.cpu().numpy()) for t in output_tuple if isinstance(t, torch.Tensor))
        elif isinstance(output, dict):
            # Convert dict of torch tensors to dict of Tensors
            output_dict = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    output_dict[key] = Tensor(value.cpu().numpy())
                else:
                    output_dict[key] = value
            
            return output_dict
        else:
            # Return as is
            return output
    
    def preprocess(self, x: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Tensor]:
        """
        Preprocess images using the processor.
        
        Args:
            x: Input images
            
        Returns:
            Processed images
        """
        # Process input
        inputs = self.processor(x, return_tensors="pt")
        
        # Convert to Tensor
        processed = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                processed[key] = Tensor(value.numpy())
            else:
                processed[key] = value
        
        return processed
    
    def save(self, path: str):
        """
        Save model and processor to disk.
        
        Args:
            path: Path to save model to
        """
        # Save model
        super().save(path)
        
        # Save processor
        self.processor.save_pretrained(f"{path}_processor")
    
    def load(self, path: str):
        """
        Load model and processor from disk.
        
        Args:
            path: Path to load model from
        """
        # Load model
        super().load(path)
        
        # Load processor
        self.processor = transformers.AutoImageProcessor.from_pretrained(f"{path}_processor")
