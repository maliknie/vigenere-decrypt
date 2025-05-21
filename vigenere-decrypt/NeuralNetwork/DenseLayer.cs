public class DenseLayer : ILayer
{
    private Matrix<float> _weights;
    private Vector<float> _bias;

    public DenseLayer(Matrix<float> weights, Vector<float> bias)
    {
        _weights = weights;
        _bias = bias;
    }

    public Vector<float> Forward(Vector<float> input)
    {
        return (input * _weights) + _bias;
    }
}
