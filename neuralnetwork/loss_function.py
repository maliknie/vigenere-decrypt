import numpy as np

# Loss Functions
class Loss:
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / y_true.shape[0]
    
    @staticmethod
    def categorical_crossentropy_derivative(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]
