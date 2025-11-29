import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class Adadelta(BaseOptimizer):
    def __init__(self, lr: float=0.0001, beta: float=0.9):
        super().__init__(lr)
        self.beta = beta
        self.grad_mov_avg = 0
        self.update_mov_avg = 0
        
    def step(self, params, grad):
        # removes the need for a learning rate by determining one automatically by means of decayed square of gradients and updates
        self.grad_mov_avg = self.beta * self.grad_mov_avg + (1 - self.beta) * grad**2
        
        # calculating update
        update = - (np.sqrt(self.update_mov_avg + 10e-8)) * grad / (np.sqrt(self.grad_mov_avg + 10e-8))
        self.update_mov_avg = self.beta * self.update_mov_avg + (1 - self.beta) * update**2
        
        return params + update