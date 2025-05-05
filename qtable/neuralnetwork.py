import torch
import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, nonlinearity_cls=None, use_dropout=False, bias=0.0):
        super(FullyConnectedLayer, self).__init__()
        layers = [nn.Linear(input_dim, output_dim)]
        if nonlinearity_cls:
            layers.append(nonlinearity_cls())
        # Initialize weights
        assert type(layers[0]) == nn.Linear
        nn.init.xavier_uniform_(layers[0].weight)
        nn.init.constant_(layers[0].bias, bias)
        self._nn = nn.Sequential(*layers)

    def forward(self, x):
        return self._nn(x)


class NeuralNetwork(nn.Module):
    def __init__(self, indim, enddim, hidden_layers, nonlinearity_cls=nn.LeakyReLU, dropout_prob=0.5, output_bias=0):
        super(NeuralNetwork, self).__init__()
        self.indim = indim
        self.enddim = enddim
        self.dropout_prob = dropout_prob

        self.layers = nn.ModuleList()

        prev_dim = indim
        for out_dim in hidden_layers:
            self.layers.append(
                FullyConnectedLayer(prev_dim, out_dim, nonlinearity_cls=nonlinearity_cls))
            prev_dim = out_dim
        if enddim:
            self.layers.append(FullyConnectedLayer(prev_dim, enddim, nonlinearity_cls=None, bias=output_bias))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Head(nn.Module):
    def __init__(self, shared, shared_outdim, outdim, bias=0):
        super().__init__()
        self.shared = shared
        self.head = nn.Linear(shared_outdim, outdim)
        nn.init.constant_(self.head.bias, bias)

    def forward(self, x):
        return self.head(self.shared(x))

def soft_update(target_net, online_net, tau):
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau*online_param.data + (1.0-tau)*target_param.data)



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

    shared = NeuralNetwork(indim=2, enddim=False, hidden_layers=[256, 256])
    head = Head(shared, 256, 1)
    x = torch.randn(5, 2)
    y = head(x)
    loss = y.mean()
    loss.backward()
    for name, param in head.named_parameters():
        print(name, param.requires_grad, param.grad.norm() if param.grad is not None else None)
