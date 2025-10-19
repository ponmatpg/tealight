from src.libvalue import Value, Neuron, Layer, MLP

def test_basic():
    x = Value(2)
    y = Value(3)
    z = x + y * y + x * y + x ** y
    z.backward()
    print(f"{z.describe()}")
    print(f"{x.describe()}")
    print(f"{y.describe()}")

def test_neuron():
    inputs = [Value(2), Value(3)]

    n = Neuron(input_size=2)
    z = n(inputs)
    print(z, n.parameters())
    z.backward()
    print(z, n.parameters())

def test_layer():
    inputs = [Value(2), Value(3)]

    l = Layer(input_size=2, output_size=1)
    zero = Value(0)
    z = sum(l(inputs), zero)
    print(z, l.parameters())
    z.backward()

def test_mlp():
    inputs = [Value(2), Value(3)]
    model  = MLP(input_size=1, output_sizes=[1])
    yhats = [model([x])[0] for x in inputs]
    ys = [Value(4), Value(6)]

    two = Value(2)
    losses = [(yi_hat - yi)**two for yi_hat, yi in zip(yhats, ys)]
    zero = Value(0)    
    total_loss = sum(losses, start=zero)
    print(total_loss, model.parameters())
    total_loss.backward()
    print(total_loss, model.parameters())

if __name__ == "__main__":
    #test_basic()
    #test_neuron()
    #test_layer()
    test_mlp()
    print("All tests passed!")