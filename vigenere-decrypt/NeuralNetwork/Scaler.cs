using System;

namespace VigenereDecrypt.NeuralNetwork
{
    public class Scaler
    {
        public float[] Mean { get; set; }
        public float[] Scale { get; set; }

        public Scaler(float[] mean, float[] scale)
        {
            Mean = mean;
            Scale = scale;
        }

        public float[] Transform(float[] input)
        {
            if (input.Length != Mean.Length || input.Length != Scale.Length)
                throw new ArgumentException("Input length does not match scaler dimensions.");

            float[] normalized = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                normalized[i] = (input[i] - Mean[i]) / Scale[i];
            }
            return normalized;
        }
    }
}
