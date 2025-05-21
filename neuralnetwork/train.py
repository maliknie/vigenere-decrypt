import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import Model
from layer import Layer
from dropout import DropoutLayer
from activation_function import Activation
from loss_function import Loss

# === 1. Load & Prepare Data ===
df = pd.read_csv('./neuralnetwork/dataset/vigenere_dataset.csv')

# Features and labels
X = df.drop('key_length', axis=1)
y = df['key_length'] - 1  # Zero-indexing for class labels

# Ensure no out-of-bounds
num_classes = y.max() + 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode labels
Y_train = np.eye(num_classes)[y_train]
Y_test = np.eye(num_classes)[y_test]

# === 2. Build Model ===
model = Model()
model.add(Layer(input_dim=X_train.shape[1], output_dim=128,
                activation_func=Activation.relu, activation_derivative=Activation.relu_derivative))
model.add(DropoutLayer(0.5))

model.add(Layer(input_dim=128, output_dim=64,
                activation_func=Activation.relu, activation_derivative=Activation.relu_derivative))
model.add(DropoutLayer(0.3))

model.add(Layer(input_dim=64, output_dim=32,
                activation_func=Activation.relu, activation_derivative=Activation.relu_derivative))
model.add(DropoutLayer(0.2))

model.add(Layer(input_dim=32, output_dim=num_classes,
                activation_func=Activation.softmax, activation_derivative=Activation.identity_derivative))  # Final softmax

# === 3. Train Model ===
model.train(
    X_train, Y_train,
    loss_fn=Loss.categorical_crossentropy,
    loss_fn_derivative=Loss.categorical_crossentropy_derivative,
    epochs=10000,
    learning_rate=0.01,
    x_val=X_test,
    y_val=Y_test,
    log_every=100
)


# === 4. Save Model and Scaler ===
with open("./neuralnetwork/trained_model/saved_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("./neuralnetwork/trained_model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# === 5. Evaluate ===
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
accuracy = np.mean(y_pred_labels == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
