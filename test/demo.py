from src.libvalue import Value, Neuron, Layer, MLP
import numpy as np

if __name__ == "__main__":
    m = 100
    X = np.random.normal(loc=0.0, scale=1.0, size=(m, 1))
    y = 3 * X + np.random.normal(loc=0.0, scale=0.1, size=(m, 1))

    model  = MLP(input_size=1, output_sizes=[1])

    inputs = [Value(x) for x in X]
    scores = [model([x]) for x in inputs]

    print(scores)