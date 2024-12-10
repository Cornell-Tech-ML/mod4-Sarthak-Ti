"""Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
from typing import Iterable, Callable

# Use this function to make a random parameter in
# your module.
def RParam(*shape: int) -> minitorch.Parameter:
    """Creates random parameters between -1 and 1 in the desired shape."""
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.

def default_log_fn(epoch: int, total_loss: float, correct: int, losses: Iterable[float]) -> None:
    """Default logging function for training

    Args:
    ----
        epoch : int
            The current epoch number
        total_loss : float
            The total loss for the epoch
        correct : int
            The number of correct predictions
        losses : List[float]
            The list of losses for each epoch

    Returns:
    -------
        None

    """
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)

class Network(minitorch.Module):
    def __init__(self, hidden_layers: int):
        super().__init__()
        #we call 3 linear layers
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """Forward pass of the network, should be the same as calling network(x)"""
        x = self.layer1.forward(x).relu()
        # print(x.shape)
        x = self.layer2.forward(x).relu()
        # print(x.shape)
        x = self.layer3.forward(x)
        # print(x.shape)
        return x.sigmoid()

class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        #now we use tensors to store the weights and biases
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)

    def forward(self, inputs: minitorch.Tensor) -> minitorch.Tensor:
        """Forward pass of the linear layer, should be the same as calling layer(x)"""
        #old implementation that does work, but not as clean
        # expanded = inputs.view(*[inputs.shape[0], 1, inputs.shape[1]])
        # mul = expanded * self.weights.value.permute(1, 0)
        # sums = mul.sum(2).view(*[inputs.shape[0], self.weights.value.shape[1]])
        # #and now we add the bias
        # biased = sums + self.bias.value

        # return biased

        #taykhoom inspired implementation
        w_reshaped = self.weights.value.view(1,*self.weights.value.shape) #the only difference is we put the 1 at the beginning!
        x_reshaped = inputs.view(*inputs.shape,1)
        b_reshaped = self.bias.value.view(1,*self.bias.value.shape)

        mul = x_reshaped * w_reshaped
        mult = mul.sum(1)

        Wx = mult.view(mult.shape[0],mult.shape[2])

        y = Wx + b_reshaped

        return y



class TensorTrain:
    def __init__(self, hidden_layers: int) -> None:
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """Run the model on a single input"""
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X: minitorch.Tensor) -> minitorch.Tensor:
        """Run the model on a batched input"""
        return self.model.forward(minitorch.tensor(X))

    def train(self, data: minitorch.Tensor, learning_rate: int, max_epochs: int = 500,
          log_fn: Callable[[int, float, int, list], None] = default_log_fn) -> None:
        """Trains a model on the given data and outputs the results based on the log_fn"""
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 3
    RATE = .1
    data = minitorch.datasets["Diag"](PTS)
    TensorTrain(HIDDEN).train(data, RATE, max_epochs=2000)
