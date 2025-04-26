"""
Stochastic Gradient Descent optimizer for the Neurenix framework.
"""

from typing import Iterable, Dict, Any, Optional

from neurenix.optim.optimizer import Optimizer
from neurenix.tensor import Tensor

class SGD(Optimizer):
    """
    Implements stochastic gradient descent (optionally with momentum).
    
    This optimizer updates parameters using the formula:
    
    If momentum is 0:
        p = p - lr * g
    
    If momentum > 0:
        v = momentum * v - lr * g
        p = p + v
    
    where p is the parameter, g is the gradient, v is the velocity,
    lr is the learning rate, and momentum is the momentum factor.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        dampening: Dampening for momentum (default: 0)
        nesterov: Enables Nesterov momentum (default: False)
    """
    
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum > 0 and dampening == 0")
        
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "dampening": dampening,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)
    
    def step(self) -> None:
        """
        Performs a single optimization step.
        
        Updates the parameters based on the current gradients.
        """
        for group in self._parameter_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    try:
                        from neurenix.binding import apply_weight_decay
                        grad = apply_weight_decay(grad, param, weight_decay)
                    except (ImportError, AttributeError):
                        grad_np = grad._numpy_data
                        param_np = param._numpy_data
                        grad_np = grad_np + weight_decay * param_np
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state.get(param, {})
                    
                    if "momentum_buffer" not in param_state:
                        # Initialize momentum buffer
                        momentum_buffer = Tensor(grad._numpy_data.copy(), device=grad.device)
                        param_state["momentum_buffer"] = momentum_buffer
                    else:
                        momentum_buffer = param_state["momentum_buffer"]
                        # Update momentum buffer
                        momentum_buffer._numpy_data = momentum * momentum_buffer._numpy_data
                        
                        if dampening != 0:
                            momentum_buffer._numpy_data += (1 - dampening) * grad._numpy_data
                        else:
                            momentum_buffer._numpy_data += grad._numpy_data
                    
                    if nesterov:
                        # Nesterov momentum
                        grad_np = grad._numpy_data + momentum * momentum_buffer._numpy_data
                    else:
                        grad_np = momentum_buffer._numpy_data
                else:
                    grad_np = grad._numpy_data
                
                # Update parameters
                param._numpy_data -= lr * grad_np
                
                # Store state
                if momentum != 0:
                    self.state[param] = param_state
