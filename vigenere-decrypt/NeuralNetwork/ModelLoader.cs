using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using MathNet.Numerics.LinearAlgebra;

namespace VigenereDecrypt.NeuralNetwork
{
    public class LayerParams
    {
        public float[][] weights { get; set; }
        public float[] bias { get; set; }
    }

    public class ScalerParams
    {
        public float[] mean { get; set; }
        public float[] scale { get; set; }
    }

    public static class ModelLoader
    {
        public static NeuralNetModel LoadFromJson(string modelPath, string scalerPath)
        {
            // Load weights and biases
            string modelJson = File.ReadAllText(modelPath);
            var layers = JsonSerializer.Deserialize<List<LayerParams>>(modelJson);

            // Load scaler
            string scalerJson = File.ReadAllText(scalerPath);
            var scalerParams = JsonSerializer.Deserialize<ScalerParams>(scalerJson);

            var model = new NeuralNetModel
            {
                Scaler = new Scaler(scalerParams.mean, scalerParams.scale)
            };

            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];
                var weightMatrix = Matrix<float>.Build.DenseOfRowArrays(layer.weights);
                var biasVector = Vector<float>.Build.Dense(layer.bias);

                model.Add(new DenseLayer(weightMatrix, biasVector));

                // Apply activation to all except last (customize if needed)
                if (i < layers.Count - 1)
                    model.Add(new ActivationLayer(ActivationLayer.ReLU));
                else
                    model.Add(new ActivationLayer(ActivationLayer.Identity));  // or softmax if needed
            }

            return model;
        }
    }
}
