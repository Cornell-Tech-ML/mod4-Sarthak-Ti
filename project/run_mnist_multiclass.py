from mnist import MNIST

import minitorch

from typing import Callable, Tuple

mndata = MNIST("project/data/")
images, labels = mndata.load_training()

BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape: int) -> minitorch.Parameter:
    """Creates random parameters between -1 and 1 in the desired shape

    Args:
        shape : tuple of ints

    Returns:
        :class:`Parameter` : random parameter of the desired shape

    """
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """Forward of the linear layer.

        Args:
            x : input tensor of shape batch x in_size

        Returns:
            output tensor of shape batch x out_size

        """
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels: int, out_channels: int, kh: int, kw: int):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input: minitorch.Tensor) -> minitorch.Tensor:
        """Forward of the convolution layer.

        Args:
            input : input tensor of shape batch x in_channels x h x w

        Returns:
            output tensor of shape batch x out_channels x h x w

        """
        out = minitorch.Conv2dFun.apply(input, self.weights.value) #out is batch x channel x h x w. Now apply bias
        return out + self.bias.value



class Network(minitorch.Module):
    """Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self):
        super().__init__()

        # For vis
        self.mid = Conv2d(1, 4, 3, 3)
        self.out = Conv2d(4, 8, 3, 3)
        self.linear1 = Linear(392, 64)
        self.linear2 = Linear(64, C)

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """Forward of the network.

        Args:
            x : input tensor of shape batch x 1 x H x W (H and W are defined above as 28)

        Returns:
            output tensor of shape batch x C which is num classes which is 10

        """
        #first we apply the first convolution
        mid = self.mid(x).relu() #size is batch x 4 x 28 x 28
        #then we apply the second convolution
        o1 = self.out(mid).relu() #size is batch x 8 x 28 x 28
        #then we apply the pooling
        # print(o1.shape)
        o2= minitorch.maxpool2d(o1, (4, 4)) #now size is batch x 8 x 7 x 7
        new_height, new_width = o2.shape[2], o2.shape[3]
        #now flatten along the last 3 dimensions
        o3 = o2.view(o2.shape[0], o2.shape[1] * new_height * new_width) #now size is batch x 392
        #now apply the linear layer
        o4 = self.linear1(o3).relu() #now size is batch x 64
        #now apply the dropout
        o5 = minitorch.dropout(o4, 0.25) #same size
        #now apply the final linear layer
        o6 = self.linear2(o5) #size is batch x 10 which is C the number of classes
        #now apply the logsoftmax
        return minitorch.nn.logsoftmax(o6, 1) #apply logsoftmax along the class dimension


def make_mnist(start: int, stop: int) -> Tuple[list, list]:
    """Load MNIST data from mndata."""
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch: int, total_loss: float, correct: int, total: int) -> None:
    """Default logging function for training, logs loss, epoch, and accuracy."""
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train: Tuple[list, list], data_val: Tuple[list,list], learning_rate: int, max_epochs: int = 500,
        log_fn: Callable[[int, float, int, int], None] = default_log_fn, early_stop: int = 5, gamma: int = 1
    ):
        """Train a model on the MNIST dataset.

        Args:
            data_train : tuple of X, y for training
            data_val : tuple of X, y for validation
            learning_rate : learning rate for SGD
            max_epochs : maximum number of epochs to train for
            log_fn : function to log progress
            early_stop : number of epochs to wait for early stopping
            gamma : learning rate decay factor

        Returns:
            None

        """
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        early_stop_loss = 100000000
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            # learning_rate *= 0.9
            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):
                if n_training_samples - example_num <= BATCH:
                    continue
                y = minitorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        x = minitorch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()
            # learning_rate *= gamma
            if early_stop is not None: #needs to not decrease for 5 epochs
                if total_loss < early_stop_loss:
                    early_stop_loss = total_loss
                else:
                    early_stop -= 1
                if early_stop == 0:
                    print('Early stopping, 5 epochs with no decrease in loss')
                    break


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    #figure out the data type of data_train
    ImageTrain().train(data_train, data_val, learning_rate=0.003, max_epochs=25, gamma=0.9, early_stop=5)
