import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim, activation_func, activation_derivative):
        """
        Fully connected layer.
        :param input_dim: number of input features
        :param output_dim: number of output neurons
        :param activation_func: activation function a(z)
        :param activation_derivative: derivative function a'(z)
        """
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation_func(self.z)
        return self.output

    def backward(self, d_out, learning_rate):
        """
        :param d_out: ∂L/∂a from the next layer or loss
        """
        d_activation = self.activation_derivative(self.z)
        d_z = d_out * d_activation  # ∂L/∂z = ∂L/∂a * a'(z)

        d_weights = np.dot(self.input.T, d_z)
        d_bias = np.sum(d_z, axis=0, keepdims=True)
        d_input = np.dot(d_z, self.weights.T)

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_input
