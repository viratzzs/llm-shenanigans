import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class RMSprop(BaseOptimizer):
    def __init__(self, lr: float=0.0001, beta: float=0.9):
        super().__init__(lr)
        self.beta = beta
        self.grad_mov_avg = 0
        
    def step(self, params, grad):
        # nearly the same as adadelta as both stemmed around the same time to deal with adagrad's superfast diminishing lr
        self.grad_mov_avg = (self.beta * self.grad_mov_avg) + (1 - self.beta) * grad**2
        new_params = params - (self.lr * grad) / (np.sqrt(self.grad_mov_avg + 10e-8))
        return new_params