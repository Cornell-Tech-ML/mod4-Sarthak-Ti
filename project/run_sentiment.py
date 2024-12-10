import random

import embeddings

import minitorch
from datasets import load_dataset
from typing import Callable, Tuple, List, Iterable, Union, Dict, Any

BACKEND = minitorch.TensorBackend(minitorch.FastOps)


def RParam(*shape: int) -> minitorch.Parameter:
    """Creates random parameters between -1 and 1 in the desired shape"""
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """Forward of the linear layer."""
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv1d(minitorch.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_width: int):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input: minitorch.Tensor) -> minitorch.Tensor:
        """Forward of the convolution layer
        input tensor of shape batch x in_channels x length
        """
        out = minitorch.Conv1dFun.apply(input, self.weights.value)
        return out + self.bias.value


class CNNSentimentKim(minitorch.Module):
    """Implement a CNN for Sentiment classification based on Y. Kim 2014.

    This model should implement the following procedure:

    1. Apply a 1d convolution with input_channels=embedding_dim
        feature_map_size=100 output channels and [3, 4, 5]-sized kernels
        followed by a non-linear activation function (the paper uses tanh, we apply a ReLu)
    2. Apply max-over-time across each feature map
    3. Apply a Linear to size C (number of classes) followed by a ReLU and Dropout with rate 25%
    4. Apply a sigmoid over the class dimension.
    """

    def __init__(
        self,
        feature_map_size: int = 100,
        embedding_size: int = 50,
        filter_sizes: list = [3, 4, 5],
        dropout: int = 0.25,
    ):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.cnn1 = Conv1d(embedding_size, feature_map_size, filter_sizes[0])
        self.cnn2 = Conv1d(embedding_size, feature_map_size, filter_sizes[1])
        self.cnn3 = Conv1d(embedding_size, feature_map_size, filter_sizes[2])
        # self.l = Linear(feature_map_size, feature_map_size)
        self.linear = Linear(feature_map_size, 1)

    def forward(self, embeddings: minitorch.Tensor) -> minitorch.Tensor:
        """Embeddings tensor: [batch x sentence length x embedding dim]"""
        x = embeddings.permute(0, 2, 1) #need sentence length to be last dimension so you can do conv over it
        #now x is shape batch x embedding dim x sentence length
        y1 = self.cnn1(x).relu() #now it's batch x feature map size x sequence length (it's same conv)
        y2 = self.cnn2(x).relu()
        y3 = self.cnn3(x).relu()
        z1 = minitorch.nn.max(y1, 2).view(y1.shape[0], y1.shape[1]) #this is the max over time
        z2 = minitorch.nn.max(y2, 2).view(y2.shape[0], y2.shape[1]) #now is shape batch x feature map size
        z3 = minitorch.nn.max(y3, 2).view(y3.shape[0], y3.shape[1])

        z = z1 + z2 + z3 #add them together, still batch x feature map size
        #let's instead concatenate them
        # z = z1.zeros((z1.shape[0], z1.shape[1] * 3))
        # z = minitorch.zeros((z1.shape[0], z1.shape[1] * 3), backend=BACKEND)
        # z[:, 0:z1.shape[1]] = z1
        # z[:, z1.shape[1]:z1.shape[1] + z2.shape[1]] = z2
        # z[:, z1.shape[1] + z2.shape[1]:] = z3

        # z = self.l(z).relu() #apply linear layer and then relu for more features
        #now apply dropout
        # z = minitorch.nn.dropout(z, 0.25, self.training) #want to apply dropout here, as it's before the features, dropout in the final layer makes no sense

        out = self.linear(z) #now apply linear layer
        out = minitorch.nn.dropout(out, 0.25, self.training) #here dropout makes no sense, but it's what the isntructions say. It also says RELU but we will definitely not apply RELU here, that actually doesn't make sense
        #however, I trained models and both ways work, so it doesn't really matter in the end. But put it here as that's what the instructions say
        return out.sigmoid().view(out.shape[0]) #apply sigmoid over the class dimension, and have to view or doesn't work for some reason...






# Evaluation helper methods
def get_predictions_array(y_true: minitorch.Tensor, model_output: minitorch.Tensor) -> list:
    """Get the predictions array for a given model output and true labels."""
    predictions_array = []
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def get_accuracy(predictions_array: list) -> float:
    """Get the accuracy for a given predictions array."""
    correct = 0
    for y_true, y_pred, logit in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array)


best_val = 0.0


def default_log_fn(
    epoch: int,
    train_loss: float,
    losses: list,
    train_predictions: list,
    train_accuracy: list,
    validation_predictions: list,
    validation_accuracy: list,
) -> None:
    """Log function for training of the sentiment model."""
    global best_val
    best_val = (
        best_val if best_val > validation_accuracy[-1] else validation_accuracy[-1]
    )
    print(f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}")
    if len(validation_predictions) > 0:
        print(f"Validation accuracy: {validation_accuracy[-1]:.2%}")
        print(f"Best Valid accuracy: {best_val:.2%}")


class SentenceSentimentTrain:
    def __init__(self, model: minitorch.Module):
        self.model = model

    def train(
        self,
        data_train: tuple,
        learning_rate: float,
        batch_size: int=10,
        max_epochs: int=500,
        data_val: tuple=None,
        log_fn: Callable[[int, float, list, list, list, list, list], None]=default_log_fn,
    ) -> None:
        """Train the model on the given data."""
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, batch_size)
            ):
                y = minitorch.tensor(
                    y_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x)
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -(prob.log() / y.shape[0]).sum()
                loss.view(1).backward()

                # Save train predictions
                train_predictions += get_predictions_array(y, out)
                total_loss += loss[0]

                # Update
                optim.step()

            # Evaluate on validation set at the end of the epoch
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                y = minitorch.tensor(
                    y_val,
                    backend=BACKEND,
                )
                x = minitorch.tensor(
                    X_val,
                    backend=BACKEND,
                )
                out = model.forward(x)
                validation_predictions += get_predictions_array(y, out)
                validation_accuracy.append(get_accuracy(validation_predictions))
                model.train()

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)
            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )
            total_loss = 0.0


def encode_sentences(
    dataset: dict, N: int, max_sentence_len: int, embeddings_lookup: Any, unk_embedding: Iterable[float], unks: set
) -> Tuple[List, List]:
    """Encode sentences in the dataset using the embeddings_lookup."""
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # pad with 0s to max sentence length in order to enable batching
        # TODO: move padding to training code
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                # use random embedding for unks
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)

    # load labels
    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(dataset: Union[Dict, List, Iterable[List]], pretrained_embeddings: Any, N_train: int, N_val: int=0) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    """Encode the sentiment data using the pretrained embeddings."""
    #  Determine max sentence length for padding
    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))
    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for i in range(pretrained_embeddings.d_emb)
    ]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")
    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    train_size = 450
    validation_size = 100
    learning_rate = 0.01
    max_epochs = 1000

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True),
        train_size,
        validation_size,
    )
    # print(X_train[0].shape, X_train[1].shape)
    # x = minitorch.tensor(
    #     X_train[:], backend=BACKEND
    # )
    # print(x.shape) #450 x 52 x 50
    # y = minitorch.tensor(
    #     y_train[:], backend=BACKEND
    # )
    # print(y.shape) #just 450
    # print(y) #is just 0 or 1
    model_trainer = SentenceSentimentTrain(
        CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
    )
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
        batch_size=50,
    )
