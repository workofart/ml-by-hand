from typing import Optional, List
from autograd.tensor import Tensor
from autograd import nn, functional, optim

# TODO: Extract the common modules out to nn.py module
from examples.transformers import (
    MultiHeadAttention,
    ResidualAddAndNorm,
    FeedForward,
    get_lr,
)
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.data import load_data, DataLoader
from autograd.text import utils as text_utils
import logging
import os
import numpy as np
from tqdm import tqdm


class GPT1(nn.Module):
    """
    GPT-1
    Paper: Improving Language Understanding by Generative Pre-Training
    https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_attention_heads,
        max_seq_len,
        dropout_prob,
        num_decoder_layers,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.sublayers = nn.ModuleList(
            [
                DecoderSublayer(
                    hidden_size=hidden_size,
                    ff_hidden_size=hidden_size * 4,
                    num_attention_heads=num_attention_heads,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(self, tokens, mask: Optional[Tensor]):
        """
        Following the same notation in the original paper
        Section 3.1 Unsupervised pre-training
        """
        batch_size, seq_len = tokens.shape
        positions = np.arange(seq_len)  # shape (seq_len,)
        positions = np.tile(positions, (batch_size, 1))  # shape (batch, seq_len)

        token_embedding = self.token_embedding(
            tokens
        )  # shape: (batch, seq_len, hidden_dim)
        position_embedding = self.position_embedding(
            positions
        )  # shape: (batch, seq_len, hidden_dim)
        h_0 = self.dropout(token_embedding + position_embedding)

        for sublayer in self.sublayers:
            h_0 = sublayer(h_0, mask)

        output = self.layer_norm(h_0)
        output = output @ self.token_embedding.parameters["weight"].T
        return functional.softmax(output)


class DecoderSublayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        ff_hidden_size=2048,
        dropout_prob=0.1,
        num_attention_heads=2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.multi_head_attention = MultiHeadAttention(
            hidden_size=hidden_size, num_heads=num_attention_heads
        )
        self.add_and_norm1 = ResidualAddAndNorm(hidden_size)
        self.add_and_norm2 = ResidualAddAndNorm(hidden_size)
        self.feedforward = FeedForward(
            fc_input_size=hidden_size,
            hidden_size=ff_hidden_size,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        x = self.add_and_norm1(
            x, lambda x_: self.multi_head_attention(x_, x_, x_, mask=mask)
        )
        x = self.add_and_norm2(x, self.feedforward)
        return x


# TODO: consolidate this with the transformers one
def inference(
    model: nn.Module,
    bpe: BytePairEncoder,
    start_tokens: List[str],
    max_length: int = 50,
    temperature: float = 1.0,
) -> str:
    """
    Peform model inference, usually for evaluation purposes.
    We will continously feed the model's generated tokens back to the model to generate
    the next token (next token is conditioned on all previously generated token).

    Args:
        model (nn.Module): The transformer model
        start_tokens (list[str]): The list of start tokens, usually ["<SOS>"]
        max_length (int, optional): The maximum length of tokens to run. Defaults to 50.
        temperature (float, optional): The amount of exploration/randomness to model output to be. Defaults to 1.0.
        > 1.0 more random
        < 1.0 less random

    Returns:
        list[str]: The list of tokens output from the model
    """
    model.eval()
    generated = [bpe._unicode_to_int_vocab[t.encode("utf-8")] for t in start_tokens]
    for _ in range(max_length):
        seq_len = len(generated)

        # "generated" is a list of integers, each int for each token
        cur_input = np.array([generated])

        causal_mask = text_utils.create_causal_mask(seq_len, 1)

        # Create causal + pad mask for this partial sequence
        # Then run model(...) with encoder_output if necessary
        probs = model(cur_input, causal_mask)
        # probs has shape (batch_size=1, seq_len, vocab_size)
        # We only care about the distribution over the last token:
        dist = probs.data[0, -1]  # shape (vocab_size,)

        # Apply temperature scaling: $$p_{i}^{(1/T)}$$
        if temperature != 1.0:
            dist = dist ** (1.0 / temperature)

        # Re-normalize the distribution (so that sum=1)
        dist_sum = np.sum(dist)
        if dist_sum <= 1e-15:
            # If the distribution collapses numerically, fall back to argmax
            next_token_id = np.argmax(dist)
        else:
            dist /= dist_sum
            # Sample from this scaled distribution
            next_token_id = np.random.choice(len(dist), p=dist)

        generated.append(next_token_id)
        # TODO: Possibly break if next_token_id is <eos> or similar
    return bpe.decode(generated)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # Based on the paper
    # Section 4.1 Setup - Model Specifications
    # TODO: parse the hyperparams from CLI
    HYPERPARAMS = {
        "num_epochs": 30,
        "warmup_steps": 1000,
        "num_attention_heads": 12,  # 12
        "d_model": 144,  # 768, must be divisible by num_attention_heads
        "batch_size": 64,  # 64
        "dropout_prob": 0.1,
        "seq_len": 128,  # 512
        "num_decoder_layers": 12,
        "eval_iters": 16,
    }

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "examples/tinyshakespeare.txt"

    data = load_data(url, filename)
    logger.info(f"Length of entire dataset: {len(data)}")

    # Create the vocabulary first
    bpe = BytePairEncoder(num_merges=3000, vocab_file_path="vocab.pkl")
    vocab, idx2word = bpe.train_vocabulary(data, overwrite_saved_file=False)

    # Now encode the subset of data
    logger.info("Encoding the new data...")
    data = data.split("\n\n")

    if os.path.exists("bpe_mini_shakespeare.npz"):
        logger.info("Found existing encoded data, loading it...")
        with np.load("bpe_mini_shakespeare.npz", allow_pickle=True) as npz_data:
            encoded_data = npz_data.get("arr_0")[:50000]
    else:
        logger.info("Encoding the new data...")
        encoded_data = np.array(bpe.encode("<|endoftext|>".join(data)))
        np.savez_compressed("bpe_mini_shakespeare.npz", encoded_data)
        logger.info("Saved encoded data to bpe_mini_shakespeare.npz")

    pad_idx = vocab[b"<PAD>"]
    logger.info(
        f"Vocabulary size: {len(vocab)}, encoded_data length: {len(encoded_data)}"
    )
    logger.info(f"Data: {data[:3]}, Encoded_Data: {encoded_data[:50]}")

    # encoded_data is a list of integers without the concept of samples
    n = int(len(encoded_data) * 0.9)
    train_data, test_data = encoded_data[:n], encoded_data[n:]

    model = GPT1(
        vocab_size=len(vocab),
        hidden_size=HYPERPARAMS["d_model"],
        num_attention_heads=HYPERPARAMS["num_attention_heads"],
        dropout_prob=HYPERPARAMS["dropout_prob"],
        max_seq_len=int(HYPERPARAMS["seq_len"] * 1.5),
        num_decoder_layers=HYPERPARAMS["num_decoder_layers"],
    )
    model.train()
    logger.info(f"Model parameters: {model.num_parameters()}")

    optimizer = optim.Adam(model.parameters, lr=0)
    train_data_loader = DataLoader(
        train_data,
        vocab,
        HYPERPARAMS["batch_size"],
        HYPERPARAMS["seq_len"],
        shuffle=True,
        pad_idx=pad_idx,
    )
    test_data_loader = DataLoader(
        test_data,
        vocab,
        HYPERPARAMS["batch_size"] // 4,
        HYPERPARAMS["seq_len"],
        shuffle=True,
        pad_idx=pad_idx,
    )
    step_count = 0

    for epoch in range(HYPERPARAMS["num_epochs"]):
        epoch_loss = 0.0
        train_data_loader.on_epoch_start()

        for x, y, _, __, causal_mask in tqdm(
            train_data_loader, desc="Step", leave=False
        ):
            step_count += 1
            lr = get_lr(step_count, HYPERPARAMS["d_model"], HYPERPARAMS["warmup_steps"])
            optimizer.lr = lr
            optimizer.zero_grad()

            # pred_probs is (batch_size, sequence_len, vocabulary_size)
            # No need the initial <SOS> token for the decoder input
            pred_prob = model(
                x,
                causal_mask,
            )
            # y has shape (batch_size, seq_len) and is already a shifted sequence
            # compared to x
            loss = functional.sparse_cross_entropy(
                pred_prob, y, pad_idx=pad_idx, label_smoothing=0.1
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().data

        if epoch % max(1, (HYPERPARAMS["num_epochs"] // 10)) == 0:
            model.eval()
            test_data_loader.on_epoch_start()
            test_loss = 0

            for _ in tqdm(
                range(HYPERPARAMS["eval_iters"]), desc="Test Evaluation", leave=False
            ):
                x, y, _, __, causal_mask = next(iter(test_data_loader))
                pred_prob = model(
                    x,
                    causal_mask,
                )
                loss = functional.sparse_cross_entropy(
                    pred_prob, y, pad_idx=pad_idx, label_smoothing=0.0
                )
                test_loss += loss.detach().data

            pred_tokens = inference(
                model,
                bpe,
                start_tokens=["All"],  # Dummy token to start the generation
                max_length=int(HYPERPARAMS["seq_len"] * 1.1),
                temperature=1.0,
            )
            model.train()

            logger.warning(
                f"\nEpoch {epoch} | Train Loss: {epoch_loss / len(train_data_loader):.2f} "
                f"| Test Loss: {test_loss / HYPERPARAMS['eval_iters']:.2f}"
            )
            prediction_string = "\n".join(pred_tokens.split("<|endoftext|>"))
            logger.info(f"Prediction:\n{prediction_string}")
