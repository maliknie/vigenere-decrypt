import pickle
import json
import numpy as np
from activation_function import Activation
from dropout import DropoutLayer
from loss_function import Loss
from model import Model

# Make sure you set neuralnetwork directory as the working directory


OUTPUT_DIR = ""

def convert_model_to_json(model_pkl_path, scaler_pkl_path,
                          out_model_json=OUTPUT_DIR+'model_params.json',
                          out_scaler_json=OUTPUT_DIR+'scaler_params.json'):
    with open(model_pkl_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_pkl_path, 'rb') as f:
        scaler = pickle.load(f)

    layers_json = []
    for layer in model.layers:
        if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
            layers_json.append({
                'weights': layer.weights.tolist(),
                'bias': layer.bias.flatten().tolist()
            })

    with open(out_model_json, 'w') as f:
        json.dump(layers_json, f, indent=2)

    scaler_json = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }

    with open(out_scaler_json, 'w') as f:
        json.dump(scaler_json, f, indent=2)

    print("Model and scaler exported to JSON.")

if __name__ == "__main__":
    convert_model_to_json(
        model_pkl_path='./trained_model/saved_model_93.pkl',
        scaler_pkl_path='./trained_model/scaler.pkl'
    )
