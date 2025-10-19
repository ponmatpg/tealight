from src.libvalue import Value, Neuron, Layer, MLP
import numpy as np

if __name__ == "__main__":
    zero = Value(0)    
    two = Value(2)
    X = [1,2,3]

    model  = MLP(input_size=1, output_sizes=[1])

    inputs = [Value(x) for x in X]
    ys = [Value(3 * x + 0.5) for x in X]

    def get_loss():
        yhats = [model([x])[0] for x in inputs]
        losses = [(yi_hat - yi)**two for yi_hat, yi in zip(yhats, ys)]
        return sum(losses, start=zero)

    lr = 0.01
    for k in range(500):
        total_loss = get_loss()
        model.zero_grad()
        total_loss.backward()    
        for p in model.parameters():
            p.step(lr)

        print(total_loss, model.parameters())