from src.libvalue import Value, Neuron

def test_basic():
    x = Value(2)
    y = Value(3)
    z = x + y * y + x * y + x ** y
    z.backward()
    print(f"{z.describe()}")
    print(f"{x.describe()}")
    print(f"{y.describe()}")

def test_neuron():
    inputs = [Value(2)]

    n = Neuron(input_size=1)
    z = n(inputs)
    print(z, n.parameters())
    z.backward()
    print(z, n.parameters())


if __name__ == "__main__":
    #test_basic()
    test_neuron()
    print("All tests passed!")