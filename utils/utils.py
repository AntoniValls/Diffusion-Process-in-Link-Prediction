import numpy as np


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.min_validation_score = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def early_stop_score(self, validation_score):
        if validation_score > self.min_validation_score:
            self.min_validation_score = validation_score
            self.counter = 0
        elif validation_score > (self.min_validation_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

