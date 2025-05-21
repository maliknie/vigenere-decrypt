public class ActivationLayer : ILayer
{
    private Func<float, float> _activationFunc;

    public ActivationLayer(Func<float, float> activationFunc)
    {
        _activationFunc = activationFunc;
    }

    public Vector<float> Forward(Vector<float> input)
    {
        return input.Map(_activationFunc);
    }

    public static float ReLU(float x) => MathF.Max(0, x);
    public static float Identity(float x) => x;
    public static float Softmax(float x) => throw new NotImplementedException(); // handled differently
}
