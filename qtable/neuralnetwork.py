import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, nonlinearity_cls=None, use_dropout=False, dropout_prob=0.5):
        super(FullyConnectedLayer, self).__init__()
        layers = [nn.Linear(input_dim, output_dim)]
        if nonlinearity_cls:
            layers.append(nonlinearity_cls())
        if use_dropout:
            # tensorflow vs pytorch
            layers.append(nn.Dropout(p=1.0 - dropout_prob))
        # Initialize weights
        assert type(layers[0]) == nn.Linear
        nn.init.normal_(layers[0].weight)
        nn.init.constant_(layers[0].bias, 1.0)
        self._nn = nn.Sequential(*layers)

    def forward(self, x):
        return self._nn(x)


class NeuralNetwork(nn.Module):
    def __init__(self, indim, enddim, hidden_layers, nonlinearity_cls=nn.ReLU, use_dropout=False, dropout_prob=0.5):
        super(NeuralNetwork, self).__init__()
        self.indim = indim
        self.enddim = enddim
        self.dropout_prob = dropout_prob

        self.layers = nn.ModuleList()

        prev_dim = indim
        for out_dim in hidden_layers:
            self.layers.append(
                FullyConnectedLayer(prev_dim, out_dim, nonlinearity_cls=nonlinearity_cls,
                                    use_dropout=use_dropout, dropout_prob=dropout_prob)
            )
            prev_dim = out_dim
        self.layers.append(FullyConnectedLayer(prev_dim, enddim, nonlinearity_cls=None))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Example usage:
if __name__ == "__main__":
    # Create a network with input dimension 784, output dimension 10
    # and two hidden layers with 128 and 64 neurons
    model = NeuralNetwork(indim=784, enddim=10, hidden_layers=[128, 64],
                         nonlinearity_cls=nn.ReLU, use_dropout=True)

    # Example input tensor
    x = torch.randn(32, 784)  # Batch size of 32

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [32, 10]

    # Set to evaluation mode (disables dropout)
    model.eval()
    eval_output = model(x)

    # Set back to training mode with different dropout probability
    model.train()
