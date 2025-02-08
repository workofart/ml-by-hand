try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np
from autograd import functional, nn, optim, tensor
from autograd.text.utils import create_vocabulary, text_to_one_hot_and_sparse
from autograd.tools.config_schema import GenericTrainingConfig
from autograd.tools.data import SimpleDataLoader, load_data
from autograd.tools.trainer import SimpleTrainer

"""
Sequence-to-sequence model

> DNNs can only be applied to problems whose inputs and targets can be sensibly encoded with vectors of fixed dimensionality. It is a significant limitation, since many important problems are best expressed with sequences whose lengths are not known a-priori. [Quoted from paper below]

This model architecture addresses the shortcoming above, allowing for variable output sequence length (e.g. summarization, image captioning). This is achieved by "rolling over" time steps, allowing the network to read an input one step at a time, accumulate an internal representation, and then decode step by step.

Paper: https://arxiv.org/abs/1409.3215
"""


class Seq2Seq(nn.Module):
    """
    A sequence-to-sequence (Seq2Seq) model that encodes an input sequence and then decodes
    it to produce a variable-length output sequence.

    The model consists of:
      - A word embedding layer that projects input tokens to a higher-dimensional space.
      - An encoder LSTM block that compresses the embedded input to a fixed-size hidden state.
      - A decoder LSTM block that is initialized with the encoder's final hidden state and produces
        a sequence of outputs.
      - A final linear layer that projects the decoder hidden state to logits over the input vocabulary.

    Attributes:
        word_embedding (nn.Linear): Projects input tokens to the embedding space.
        encoder (nn.LongShortTermMemoryBlock): Processes the embedded input sequence.
        decoder (nn.LongShortTermMemoryBlock): Generates output sequence from the encoder state.
        fc (nn.Linear): Projects the decoder output to logits.
        max_output_len (int): Maximum number of decoding time steps.
    """

    def __init__(
        self, input_size, word_embed_size, hidden_size, max_output_len: int = 30
    ):
        """
        Initialize the Seq2Seq model.

        Args:
            input_size (int): Dimensionality of the input (vocabulary size).
            word_embed_size (int): Size of the word embedding.
            hidden_size (int): Number of hidden units in the encoder and decoder LSTM blocks.
            max_output_len (int): Maximum length of the output sequence. Defaults to 30.
        """
        super().__init__()
        self.word_embedding = nn.Linear(
            input_size=input_size, output_size=word_embed_size
        )
        self.encoder = nn.LongShortTermMemoryBlock(
            input_size=word_embed_size, hidden_size=hidden_size, output_size=None
        )  # compress to shape (batch_size, hidden_size)
        self.decoder = nn.LongShortTermMemoryBlock(
            input_size=word_embed_size, hidden_size=hidden_size, output_size=None
        )  # decompress to shape (batch_size, hidden_size)

        # Final hidden layer to output a probability distribution over possible vocabulary tokens
        self.fc = nn.Linear(hidden_size, output_size=input_size)

        self.max_output_len = max_output_len

    def forward(self, x):
        """
        Perform the forward pass of the Seq2Seq model.

        The method embeds the input, encodes it using the encoder LSTM, and then iteratively decodes
        a sequence by running the decoder for a fixed number of time steps. At each time step, the decoder's
        output is projected to logits via a linear layer.

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: A tensor of shape (batch_size, max_output_len, input_size) containing the output logits.
        """
        x = tensor.Tensor(x)
        output = []
        x = functional.relu(self.word_embedding(x))
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
            # pred_token_indices = np.argmax(functional.softmax(logits).data, axis=1)
            # if np.all(pred_token_indices == 0):
            #     break
        return tensor.Tensor.stack(output, axis=1)


def parse_data_into_xy(data: np.ndarray):
    """
    Parse raw input data into source and target sequences for the Seq2Seq model.

    This function expects the input data to have the following columns:
        - URL, title, summary, article, step headers.
    It processes the data by:
        - Printing the shape of the data.
        - Adding start-of-sequence ("<SOS>") and end-of-sequence ("<EOS>") markers to the summaries.
        - Reversing the order of the articles to potentially shorten the gradient path during training.

    Args:
        data (np.ndarray): A numpy array containing the dataset, where the summary is in column index 2 and the article is in column index 3.

    Returns:
        Tuple[List[str], List[str]]:
            - articles: A list of reversed article strings.
            - summaries: A list of summaries with "<SOS>" and "<EOS>" markers.
    """
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
    """
    Main function for training the Seq2Seq model on a text summarization task.

    The function performs the following steps:
      1. Loads training and test data from parquet files (downloaded if not present).
      2. Parses the data into source and target sequences using parse_data_into_xy.
      3. Creates a vocabulary from the combined source and target texts.
      4. Converts the text sequences into one-hot encoded features and integer index matrices.
      5. Creates SimpleDataLoader objects for training and testing.
      6. Configures the training parameters via a GenericTrainingConfig.
      7. Creates a SimpleTrainer with the Seq2Seq model and trains the model.

    No value is returned; training progress and evaluation metrics are logged.
    """
    train_data_url = "https://huggingface.co/datasets/d0rj/wikisum/resolve/main/data/train-00000-of-00001-b28959cff7dcaf55.parquet"
    test_data_url = "https://huggingface.co/datasets/d0rj/wikisum/resolve/main/data/test-00000-of-00001-52a8a7cd640a9fff.parquet"

    train_filename = "training_data/wikisum_train.parquet"
    test_filename = "training_data/wikisum_test.parquet"

    train_data = load_data(train_data_url, train_filename, max_rows=1024)
    test_data = load_data(test_data_url, test_filename, max_rows=1024)

    train_X, train_y = parse_data_into_xy(train_data)
    test_X, test_y = parse_data_into_xy(test_data)

    vocab = create_vocabulary(train_X + train_y, max_features=10000)
    features, features_vocab_idx = text_to_one_hot_and_sparse(
        train_X, vocab, max_sequence_length=120
    )
    labels, labels_vocab_idx = text_to_one_hot_and_sparse(
        train_y, vocab, max_sequence_length=60
    )
    test_features, features_vocab_idx = text_to_one_hot_and_sparse(
        train_X, vocab, max_sequence_length=120
    )
    test_labels, test_labels_vocab_idx = text_to_one_hot_and_sparse(
        train_y, vocab, max_sequence_length=60
    )

    train_data_loader = SimpleDataLoader(
        features,
        labels_vocab_idx,
        batch_size=32,
        shuffle=True,
    )
    test_data_loader = SimpleDataLoader(
        test_features,
        test_labels_vocab_idx,
        batch_size=32,
        shuffle=False,
    )

    # Build a training configuration for the Seq2Seq model.
    config = GenericTrainingConfig(
        total_epochs=40,
        checkpoint_freq=20,
        model_kwargs={
            "input_size": len(vocab),
            "word_embed_size": 512,
            "hidden_size": 128,
            "max_output_len": 60,
        },
        optimizer_kwargs={"lr": 0.001},
    )

    # Create the trainer using the unified interface.
    trainer_instance = SimpleTrainer(
        model_cls=Seq2Seq,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        output_type="logits",
        config=config,
        sample_predictions=True,
    )

    trainer_instance.fit(train_data_loader, test_data_loader)


if __name__ == "__main__":
    main()
