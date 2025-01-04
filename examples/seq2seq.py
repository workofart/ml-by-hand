import numpy as np
from autograd import nn, optim, functional, tensor
from autograd.tools.trainer import Trainer
from autograd.tools.data import text_to_one_hot_and_sparse, create_vocabulary
import pyarrow.parquet as pq
import requests
import os


"""
Sequence-to-sequence model

> DNNs can only be applied to problems whose inputs and targets can be sensibly encoded with vectors of fixed dimensionality. It is a significant limitation, since many important problems are best expressed with sequences whose lengths are not known a-priori. [Quoted from paper below]

This model architecture addresses the shortcoming above, allowing for variable output sequence length (e.g. summarization, image captioning). This is achieved by "rolling over" time steps, allowing the network to read an input one step at a time, accumulate an internal representation, and then decode step by step.

Paper: https://arxiv.org/abs/1409.3215
"""


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, vocab, max_output_len: int = 30):
        super().__init__()
        self.encoder = nn.LongShortTermMemoryBlock(
            input_size=input_size, hidden_size=hidden_size, output_size=None
        )  # compress to shape (batch_size, hidden_size)
        self.decoder = nn.LongShortTermMemoryBlock(
            input_size=len(vocab), hidden_size=hidden_size, output_size=None
        )  # decompress to shape (batch_size, hidden_size)

        # Final hidden layer to output a probability distribution over possible vocabulary tokens
        self.fc = nn.Linear(hidden_size, output_size=len(vocab))

        self.vocab = vocab
        self.vocab_indices = np.array(list(vocab.keys()))
        self.max_output_len = max_output_len

    def forward(self, x):
        x = tensor.Tensor(x)

        output = []

        h_t, cell_t = self.encoder(x)

        for t in range(self.max_output_len):
            x_t = x[:, t, :]
            x_t = x_t.view((x_t.shape[0], 1, x_t.shape[1]))

            # We will initialize the decoder with the encoder's final hidden state
            h_t, cell_t = self.decoder(x=x_t, hidden_state=h_t, C_t=cell_t)

            # Get the final projection in logits format
            logits = self.fc(h_t)
            output.append(logits)

            # TODO: If all the predicted tokens in the batch are <EOS>,
            # we can stop decoding the batch
            # pred_token_indices = np.argmax(functional.softmax(logits), axis=1)
            # if np.all(self.vocab_indices[pred_token_indices] == "<EOS>"):
            #     break

        return tensor.Tensor.stack(output, axis=1)


def load_data(url, filename):
    # Check if file already exists
    if os.path.exists(filename):
        # Read the existing Parquet file into a numpy array
        table = pq.read_table(filename)
        data = table.to_pandas().to_numpy()
        return data

    # Download the file
    response = requests.get(url)

    # Save the file
    with open(filename, "wb") as f:
        f.write(response.content)

    # Read the Parquet file into a numpy array
    table = pq.read_table(filename)
    data = table.to_pandas().to_numpy()

    return data


def parse_data_into_xy(data: np.ndarray):
    # Data format
    # url, title, summary, article, step headers
    print(f"{data.shape=}")

    # Enrich the label summary with start of string and end of string markers
    summaries = [f"<SOS> {summary} <EOS>" for summary in data[:, 2]]

    # In the paper, reversing the input sentence minimizes the distance between the
    # start of the source sentence and the relevant parts in the output sentence.
    # It shortens the "path" the gradients have to traverse in time.
    articles = data[:, 3][::-1]

    return articles, summaries


def main():
    train_data_url = "https://huggingface.co/datasets/d0rj/wikisum/resolve/main/data/train-00000-of-00001-b28959cff7dcaf55.parquet"
    test_data_url = "https://huggingface.co/datasets/d0rj/wikisum/resolve/main/data/test-00000-of-00001-52a8a7cd640a9fff.parquet"

    train_filename = "examples/wikisum_train.parquet"
    test_filename = "examples/wikisum_test.parquet"

    train_data = load_data(train_data_url, train_filename)
    test_data = load_data(test_data_url, test_filename)

    train_X, train_y = parse_data_into_xy(train_data)
    test_X, test_y = parse_data_into_xy(test_data)

    vocab = create_vocabulary(train_X + train_y, max_features=6000)
    idx_to_vocab = np.array(list(vocab.keys()))
    features, _ = text_to_one_hot_and_sparse(train_X, vocab, max_sequence_length=30)
    labels, _ = text_to_one_hot_and_sparse(train_y, vocab, max_sequence_length=30)

    model = Seq2Seq(
        input_size=len(vocab),
        hidden_size=1024,
        vocab=vocab,
        max_output_len=30,
    )

    trainer = Trainer(
        model,
        loss_fn=functional.cross_entropy_with_logits,
        optimizer=optim.Adam(model.parameters, lr=0.001),
        epochs=1000,
        batch_size=128,
        output_type="logits",
    )

    trainer.fit(features, labels, idx_to_vocab)


if __name__ == "__main__":
    main()
