from typing import Any
import numpy as np

class EarlyStop:
    def __init__(self, patience) -> None:
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_model = None
        self.patience = patience
        self.counter = 0
    
    def __call__(self, model, loss, epoch):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_model = model
            self.counter = 0
        else:
            self.counter += 1

        if self.counter == self.patience:
            print(f"Loss did not improve over {self.patience} epochs, stopping early")
            return True
        
        return False
    
class RunningAverage:
    def __init__(self) -> None:
        self.avg = 0
        self.n = 0

    def __call__(self, new_value) -> Any:
        self.n += 1
        self.avg = self.avg + (new_value - self.avg)/self.n
        return self.avg
    
    def reset(self):
        self.avg = 0
        self.n = 0
    