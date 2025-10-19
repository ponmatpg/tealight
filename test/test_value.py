from src.libvalue import Value, Neuron, Layer, MLP

def test_basic():
    x = Value(2)
    y = Value(3)
    z = x + y * y + x * y + x ** y
    z.backward()
    print(f"{z.describe()}")
    print(f"{x.describe()}")
    print(f"{y.describe()}")

    n = Neuron(input_size=10)
    l = Layer(input_size=10, output_size=5)
    mlp = MLP(input_size=1, output_sizes=[1])

if __name__ == "__main__":
    test_basic()
    print("All tests passed!")