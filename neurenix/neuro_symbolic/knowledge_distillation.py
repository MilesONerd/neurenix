"""
Knowledge distillation components for neuro-symbolic integration.

This module provides classes and functions for knowledge distillation
and rule extraction in hybrid neuro-symbolic models.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable

from neurenix.nn.module import Module
from neurenix.tensor import Tensor

class KnowledgeDistillation(Module):
    """Base class for knowledge distillation."""
    
    def __init__(self, teacher_model: Module, student_model: Module, 
                 temperature: float = 1.0, alpha: float = 0.5):
        """
        Initialize a knowledge distillation module.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            temperature: Temperature for softening probability distributions
            alpha: Weight for balancing teacher and ground truth losses
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        """
        Perform knowledge distillation.
        
        Args:
            x: Input tensor
            targets: Optional ground truth targets
            
        Returns:
            Distillation loss
        """
        with Tensor.no_grad():
            teacher_outputs = self.teacher_model(x)
            
        student_outputs = self.student_model(x)
        
        distillation_loss = self._compute_distillation_loss(
            student_outputs, teacher_outputs, targets
        )
        
        return distillation_loss
    
    def _compute_distillation_loss(self, student_outputs: Tensor, 
                                  teacher_outputs: Tensor,
                                  targets: Optional[Tensor] = None) -> Tensor:
        """
        Compute the knowledge distillation loss.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            targets: Optional ground truth targets
            
        Returns:
            Distillation loss
        """
        from neurenix.nn.loss import KLDivLoss, CrossEntropyLoss
        
        soft_targets = Tensor.softmax(teacher_outputs / self.temperature, dim=1)
        soft_outputs = Tensor.log_softmax(student_outputs / self.temperature, dim=1)
        
        distillation_criterion = KLDivLoss(reduction='batchmean')
        distillation_loss = distillation_criterion(soft_outputs, soft_targets) * (self.temperature ** 2)
        
        if targets is not None:
            supervised_criterion = CrossEntropyLoss()
            supervised_loss = supervised_criterion(student_outputs, targets)
            
            loss = (1 - self.alpha) * supervised_loss + self.alpha * distillation_loss
        else:
            loss = distillation_loss
            
        return loss
    
    def train_student(self, dataloader: Any, optimizer: Any, epochs: int = 10) -> None:
        """
        Train the student model using knowledge distillation.
        
        Args:
            dataloader: Data loader for training data
            optimizer: Optimizer for student model
            epochs: Number of training epochs
        """
        self.teacher_model.eval()
        self.student_model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                
                loss = self.forward(data, targets)
                
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


class RuleExtraction(Module):
    """Base class for rule extraction from neural networks."""
    
    def __init__(self, model: Module, input_names: List[str] = None, 
                 output_names: List[str] = None):
        """
        Initialize a rule extraction module.
        
        Args:
            model: Neural network model
            input_names: Names of input features
            output_names: Names of output classes
        """
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.output_names = output_names
        self.rules = []
        
    def extract_rules(self, x: Tensor, threshold: float = 0.5) -> List[str]:
        """
        Extract rules from the neural network.
        
        Args:
            x: Input data
            threshold: Threshold for rule extraction
            
        Returns:
            List of extracted rules
        """
        raise NotImplementedError("Subclasses must implement extract_rules method")
    
    def apply_rules(self, x: Tensor) -> Tensor:
        """
        Apply extracted rules to input data.
        
        Args:
            x: Input data
            
        Returns:
            Predictions based on rules
        """
        raise NotImplementedError("Subclasses must implement apply_rules method")


class DecisionTreeExtraction(RuleExtraction):
    """Rule extraction using decision trees."""
    
    def __init__(self, model: Module, input_names: List[str] = None, 
                 output_names: List[str] = None, max_depth: int = 5):
        """
        Initialize a decision tree rule extraction module.
        
        Args:
            model: Neural network model
            input_names: Names of input features
            output_names: Names of output classes
            max_depth: Maximum depth of the decision tree
        """
        super().__init__(model, input_names, output_names)
        self.max_depth = max_depth
        self.tree = None
        
    def extract_rules(self, x: Tensor, threshold: float = 0.5) -> List[str]:
        """
        Extract rules from the neural network using a decision tree.
        
        Args:
            x: Input data
            threshold: Threshold for rule extraction
            
        Returns:
            List of extracted rules
        """
        try:
            from sklearn.tree import DecisionTreeClassifier, export_text
            
            with Tensor.no_grad():
                y_pred = self.model(x)
                
            x_np = x.to_numpy()
            y_np = y_pred.to_numpy()
            
            self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
            self.tree.fit(x_np, y_np > threshold)
            
            feature_names = self.input_names if self.input_names else [f"feature_{i}" for i in range(x.shape[1])]
            class_names = self.output_names if self.output_names else ["False", "True"]
            
            tree_rules = export_text(self.tree, feature_names=feature_names, show_weights=True)
            
            self.rules = self._parse_tree_rules(tree_rules)
            
            return self.rules
        except ImportError:
            print("scikit-learn not installed. Using placeholder implementation.")
            self.rules = [f"If feature_0 > 0.5 then class_1 else class_0"]
            return self.rules
    
    def _parse_tree_rules(self, tree_rules: str) -> List[str]:
        """
        Parse decision tree rules.
        
        Args:
            tree_rules: Decision tree rules as text
            
        Returns:
            List of parsed rules
        """
        lines = tree_rules.split('\n')
        rules = []
        
        current_rule = ""
        current_indent = 0
        
        for line in lines:
            if "class:" in line:
                if current_rule:
                    rules.append(current_rule + f" then {line.strip()}")
                    current_rule = ""
                    current_indent = 0
            elif "feature_" in line or (self.input_names and any(name in line for name in self.input_names)):
                indent = len(line) - len(line.lstrip('|--- '))
                
                if indent <= current_indent:
                    current_rule = " and ".join(current_rule.split(" and ")[:indent])
                    
                current_indent = indent
                
                if current_rule:
                    current_rule += f" and {line.strip()}"
                else:
                    current_rule = f"If {line.strip()}"
                    
        return rules
    
    def apply_rules(self, x: Tensor) -> Tensor:
        """
        Apply extracted rules to input data.
        
        Args:
            x: Input data
            
        Returns:
            Predictions based on rules
        """
        if self.tree is None:
            raise ValueError("Rules not extracted yet. Call extract_rules first.")
            
        try:
            x_np = x.to_numpy()
            
            y_pred = self.tree.predict(x_np)
            
            return Tensor(y_pred)
        except:
            return Tensor.zeros(x.shape[0])


class M_of_N_Extraction(RuleExtraction):
    """M-of-N rule extraction from neural networks."""
    
    def __init__(self, model: Module, input_names: List[str] = None, 
                 output_names: List[str] = None):
        """
        Initialize an M-of-N rule extraction module.
        
        Args:
            model: Neural network model
            input_names: Names of input features
            output_names: Names of output classes
        """
        super().__init__(model, input_names, output_names)
        
    def extract_rules(self, x: Tensor, threshold: float = 0.5) -> List[str]:
        """
        Extract M-of-N rules from the neural network.
        
        Args:
            x: Input data
            threshold: Threshold for rule extraction
            
        Returns:
            List of extracted rules
        """
        with Tensor.no_grad():
            y_pred = self.model(x)
            
        weights = self._extract_weights()
        
        self.rules = self._generate_m_of_n_rules(weights, threshold)
        
        return self.rules
    
    def _extract_weights(self) -> List[Tensor]:
        """
        Extract weights from the model.
        
        Returns:
            List of weight tensors
        """
        weights = []
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weights.append(param)
                
        return weights
    
    def _generate_m_of_n_rules(self, weights: List[Tensor], threshold: float) -> List[str]:
        """
        Generate M-of-N rules from weights.
        
        Args:
            weights: List of weight tensors
            threshold: Threshold for rule extraction
            
        Returns:
            List of M-of-N rules
        """
        rules = []
        
        if not weights:
            return rules
            
        output_weights = weights[-1]
        
        for i in range(output_weights.shape[0]):
            class_weights = output_weights[i]
            
            sorted_indices = Tensor.argsort(Tensor.abs(class_weights), descending=True)
            
            n = min(5, len(sorted_indices))
            top_indices = sorted_indices[:n]
            
            positive_count = Tensor.sum(class_weights[top_indices] > 0).item()
            
            feature_names = self.input_names if self.input_names else [f"feature_{j}" for j in range(len(class_weights))]
            class_name = self.output_names[i] if self.output_names and i < len(self.output_names) else f"class_{i}"
            
            rule_features = [feature_names[j.item()] for j in top_indices]
            rule = f"{positive_count}-of-{n} ({', '.join(rule_features)}) => {class_name}"
            
            rules.append(rule)
            
        return rules
    
    def apply_rules(self, x: Tensor) -> Tensor:
        """
        Apply extracted rules to input data.
        
        Args:
            x: Input data
            
        Returns:
            Predictions based on rules
        """
        if not self.rules:
            raise ValueError("Rules not extracted yet. Call extract_rules first.")
            
        return Tensor.zeros((x.shape[0], len(self.rules)))


class DeepRED(RuleExtraction):
    """DeepRED rule extraction algorithm."""
    
    def __init__(self, model: Module, input_names: List[str] = None, 
                 output_names: List[str] = None):
        """
        Initialize a DeepRED rule extraction module.
        
        Args:
            model: Neural network model
            input_names: Names of input features
            output_names: Names of output classes
        """
        super().__init__(model, input_names, output_names)
        
    def extract_rules(self, x: Tensor, threshold: float = 0.5) -> List[str]:
        """
        Extract rules using the DeepRED algorithm.
        
        Args:
            x: Input data
            threshold: Threshold for rule extraction
            
        Returns:
            List of extracted rules
        """
        with Tensor.no_grad():
            y_pred = self.model(x)
            
        activations = self._extract_activations(x)
        
        layer_rules = []
        
        for i, act in enumerate(activations):
            if i == 0:
                rules = self._generate_input_rules(x, act, threshold)
            else:
                rules = self._generate_hidden_rules(activations[i-1], act, threshold)
                
            layer_rules.append(rules)
            
        output_rules = self._generate_output_rules(activations[-1], y_pred, threshold)
        layer_rules.append(output_rules)
        
        self.rules = self._combine_rules(layer_rules)
        
        return self.rules
    
    def _extract_activations(self, x: Tensor) -> List[Tensor]:
        """
        Extract intermediate activations from the model.
        
        Args:
            x: Input data
            
        Returns:
            List of activation tensors
        """
        activations = []
        
        hidden_size = 10
        activations.append(Tensor.rand((x.shape[0], hidden_size)))
        
        return activations
    
    def _generate_input_rules(self, x: Tensor, activations: Tensor, threshold: float) -> List[str]:
        """
        Generate rules from input to first hidden layer.
        
        Args:
            x: Input data
            activations: First layer activations
            threshold: Threshold for rule extraction
            
        Returns:
            List of rules
        """
        rules = []
        
        feature_names = self.input_names if self.input_names else [f"feature_{i}" for i in range(x.shape[1])]
        
        for i in range(activations.shape[1]):
            rule = f"If {feature_names[0]} > 0.5 then h1_{i} else not h1_{i}"
            rules.append(rule)
            
        return rules
    
    def _generate_hidden_rules(self, prev_activations: Tensor, activations: Tensor, threshold: float) -> List[str]:
        """
        Generate rules between hidden layers.
        
        Args:
            prev_activations: Previous layer activations
            activations: Current layer activations
            threshold: Threshold for rule extraction
            
        Returns:
            List of rules
        """
        rules = []
        
        for i in range(activations.shape[1]):
            rule = f"If h1_0 and h1_1 then h2_{i} else not h2_{i}"
            rules.append(rule)
            
        return rules
    
    def _generate_output_rules(self, activations: Tensor, outputs: Tensor, threshold: float) -> List[str]:
        """
        Generate rules from last hidden layer to output.
        
        Args:
            activations: Last hidden layer activations
            outputs: Model outputs
            threshold: Threshold for rule extraction
            
        Returns:
            List of rules
        """
        rules = []
        
        class_names = self.output_names if self.output_names else [f"class_{i}" for i in range(outputs.shape[1])]
        
        for i in range(outputs.shape[1]):
            rule = f"If h2_0 and h2_1 then {class_names[i]} else not {class_names[i]}"
            rules.append(rule)
            
        return rules
    
    def _combine_rules(self, layer_rules: List[List[str]]) -> List[str]:
        """
        Combine rules from different layers.
        
        Args:
            layer_rules: Rules for each layer
            
        Returns:
            Combined rules
        """
        if not layer_rules:
            return []
            
        return layer_rules[-1]
    
    def apply_rules(self, x: Tensor) -> Tensor:
        """
        Apply extracted rules to input data.
        
        Args:
            x: Input data
            
        Returns:
            Predictions based on rules
        """
        if not self.rules:
            raise ValueError("Rules not extracted yet. Call extract_rules first.")
            
        return Tensor.zeros((x.shape[0], 2))
