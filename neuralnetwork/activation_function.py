import numpy as np

# Activation Functions
class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    import numpy as np

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    @staticmethod
    def softmax_jacobian(x):
        """
        Computes the Jacobian matrix for the softmax function for each sample in the batch.
        Input:
            x: numpy array of shape (batch_size, num_classes)
        Output:
            jacobians: numpy array of shape (batch_size, num_classes, num_classes)
        """
        softmax_vals = Activation.softmax(x)
        batch_size, num_classes = softmax_vals.shape
        jacobians = np.zeros((batch_size, num_classes, num_classes))

        for b in range(batch_size):
            s = softmax_vals[b].reshape(-1, 1)  # Column vector
            jacobians[b] = np.diagflat(s) - s @ s.T  # J = diag(s) - s * s^T

        return jacobians
    
    @staticmethod
    def identity_derivative(x):
        return 1
