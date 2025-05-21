import numpy as np

class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True  # This will be toggled by the model

    def forward(self, input_data):
        if self.training:
            self.mask = (np.random.rand(*input_data.shape) > self.dropout_rate).astype(float)
            return input_data * self.mask / (1.0 - self.dropout_rate)
        else:
            return input_data  # No dropout at inference time

    def backward(self, d_out, learning_rate):
        return d_out * self.mask / (1.0 - self.dropout_rate)
