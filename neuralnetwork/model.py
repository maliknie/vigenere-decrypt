import numpy as np
import pickle

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set_training(self, mode=True):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = mode

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def train(self, x_train, y_train, loss_fn, loss_fn_derivative, epochs=10, learning_rate=0.01, x_val=None, y_val=None, log_every=500):
        for epoch in range(epochs):
            predictions = self.forward(x_train)
            loss = loss_fn(y_train, predictions)
            loss_grad = loss_fn_derivative(y_train, predictions)
            self.backward(loss_grad, learning_rate)

            if (epoch + 1) % log_every == 0 and x_val is not None:
                val_pred = self.forward(x_val)
                val_pred_labels = np.argmax(val_pred, axis=1)
                y_val_labels = np.argmax(y_val, axis=1)
                accuracy = np.mean(val_pred_labels == y_val_labels)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Val Accuracy: {accuracy * 100:.2f}%")

                # Save checkpoint
                with open("saved_model.pkl", "wb") as f:
                    pickle.dump(self, f)
                print("Model checkpoint saved.")

        print(f"Final Epoch - Loss: {loss:.4f}")


    def predict(self, x):
        self.set_training(False)
        return self.forward(x)
