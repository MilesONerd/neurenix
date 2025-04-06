"""
Adam optimizer for the Neurenix framework.
"""

from typing import Iterable, Dict, Any, Optional
import math

from neurenix.optim.optimizer import Optimizer
from neurenix.tensor import Tensor

class Adam(Optimizer):
    """
    Implements the Adam optimization algorithm.
    
    Adam is an adaptive learning rate optimization algorithm designed
    specifically for training deep neural networks.
    
    The algorithm is described in the paper:
    "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014).
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients used for computing running averages of gradient
              and its square (default: (0.9, 0.999))
        eps: Term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        amsgrad: Whether to use the AMSGrad variant (default: False)
    """
    
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 0 or 1: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)
    
    def step(self) -> None:
        """
        Performs a single optimization step.
        
        Updates the parameters based on the current gradients.
        """
        for group in self._parameter_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            amsgrad = group["amsgrad"]
            
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Get or initialize state
                param_state = self.state.get(param, {})
                
                # Initialize state if needed
                if len(param_state) == 0:
                    param_state["step"] = 0
                    # Exponential moving average of gradient values
                    param_state["exp_avg"] = Tensor(
                        0 * grad._numpy_data,
                        device=grad.device
                    )
                    # Exponential moving average of squared gradient values
                    param_state["exp_avg_sq"] = Tensor(
                        0 * grad._numpy_data,
                        device=grad.device
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        param_state["max_exp_avg_sq"] = Tensor(
                            0 * grad._numpy_data,
                            device=grad.device
                        )
                
                # Update step count
                param_state["step"] += 1
                step = param_state["step"]
                
                # Get state variables
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]
                
                # Apply weight decay
                if weight_decay != 0:
                    try:
                        from neurenix.binding import apply_weight_decay
                        grad = apply_weight_decay(grad, param, weight_decay)
                    except (ImportError, AttributeError):
                        grad_np = grad._numpy_data
                        param_np = param._numpy_data
                        grad_np = grad_np + weight_decay * param_np
                
                # Update biased first moment estimate
                exp_avg._numpy_data = beta1 * exp_avg._numpy_data + (1 - beta1) * grad._numpy_data
                
                # Update biased second raw moment estimate
                exp_avg_sq._numpy_data = beta2 * exp_avg_sq._numpy_data + (1 - beta2) * (grad._numpy_data ** 2)
                
                if amsgrad:
                    max_exp_avg_sq = param_state["max_exp_avg_sq"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    import numpy as np
                    max_exp_avg_sq._numpy_data = np.maximum(max_exp_avg_sq._numpy_data, exp_avg_sq._numpy_data)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq._numpy_data ** 0.5) + eps
                else:
                    denom = (exp_avg_sq._numpy_data ** 0.5) + eps
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                param._numpy_data -= step_size * exp_avg._numpy_data / denom
                
                # Store state
                self.state[param] = param_state
