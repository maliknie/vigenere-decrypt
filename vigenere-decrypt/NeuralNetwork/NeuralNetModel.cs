public class NeuralNetModel
{
    private List<ILayer> _layers = new();

    public void Add(ILayer layer)
    {
        _layers.Add(layer);
    }

    public Vector<float> Predict(Vector<float> input)
    {
        foreach (var layer in _layers)
        {
            input = layer.Forward(input);
        }
        return input;
    }

    public int PredictClass(Vector<float> input)
    {
        var output = Predict(input);
        return output.MaximumIndex();
    }
}
