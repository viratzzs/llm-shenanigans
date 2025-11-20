import numpy as np
import json
import os

from bgd import BatchGradientDescent
from base_optimizer import BaseOptimizer
from loss_functions import LossFunction, Rosenbrock

class Simulator:
    def __init__(self, optimizer: BaseOptimizer, loss_fn: LossFunction, params: np.ndarray, steps: int):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.params = params
        self.steps = steps
        self.trajectory = []
    
    def run(self):
        #loss = self.loss_fn.evaluate(*self.params)
        loss = self.loss_fn.evaluate(self.params)
        #self.trajectory.append((*self.params, loss))
        self.trajectory.append((self.params.tolist(), loss.tolist() if isinstance(loss, np.ndarray) else loss))
        
        for _ in range(self.steps):
        #    grad = self.loss_fn.grad(*self.params)  # unpack array into positional args
            grad = self.loss_fn.grad(self.params)
        #    self.params = self.optimizer.step(self.params, grad) # update params with respective optimizer
            self.params = self.optimizer.step(self.params, grad)
        #    loss = self.loss_fn.evaluate(*self.params) # evaluate loss with updated params
            loss = self.loss_fn.evaluate(self.params)
        #    self.trajectory.append((*self.params, loss))
            self.trajectory.append((self.params.tolist(), loss.tolist() if isinstance(loss, np.ndarray) else loss))

    def export_trajectory(self, path: str = "trajectories/gd.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(self.trajectory, open(path, "w"))

def main():
    #init_pos = np.array([-0.5, 2.0])
    #init_pos = np.random.rand(2)
    params = np.random.rand(2,20)
    print(f"Starting position: {params}")
    s1 = Simulator(BatchGradientDescent(), Rosenbrock(baby_mode=False), params, 7500)
    s1.run()
    s1.export_trajectory()
    
if __name__ == "__main__":
    main()