from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], weight_decay: float, eps: float) -> None:
        defaults = {"lr_hp": lr, "beta1_hp": betas[0], "beta2_hp": betas[1], "lambda_hp": weight_decay, "epsilon_hp": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr_hp"]
            beta1 = group["beta1_hp"]
            beta2 = group["beta2_hp"]
            lambda_hp = group["lambda_hp"]
            epsilon = group["epsilon_hp"]


            for p in group["params"]:
                if p.grad is None:
                    continue

                # the state vector should have everything we need for one param!
                state = self.state[p]
                grad = p.grad.data
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)  # First moment vector
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)  # Second moment vector
                if "t" not in state:
                    state["t"] = 1  # Time step

                m = state["m"]
                v = state["v"]
                t = state["t"]

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)
                alpha_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data -= alpha_t * m / ((torch.sqrt(v)) + epsilon)
                p.data -= lr * lambda_hp * p.data

                state["m"] = m
                state["v"] = v
                state["t"] += 1

        return loss
