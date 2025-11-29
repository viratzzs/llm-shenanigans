import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class Nadam(BaseOptimizer):
    """
    Love child of nesterov and adam cursed to obscurity
    """
    def __init__(self, lr: float = 0.0001, beta1: float=0.9, beta2: float=0.999):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = 0
        self.velocity = 0
        
    def step(self, params, grad, cur_step):
        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * grad
        self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * grad**2
        
        # here we go again
        m1 = self.momentum / (1 - self.beta1 ** cur_step)
        m2 = self.velocity / (1 - self.beta2 ** cur_step)
        
        m1_bar = self.beta1 * m1 + ((1 - self.beta1) * grad) / (1 - self.beta1**cur_step)
        
        new_params = params - (self.lr * m1_bar) / (np.sqrt(m2) + 1e-8)
        return new_params