import numpy as np

from autograd.tools.data import load_data
from autograd.text import utils as text_utils
from autograd.tensor import Tensor
from autograd import nn, functional, optim

np.random.seed(1337)


class Transformer(nn.Module):
    """
    Implements the Paper "Attention is All You Need"
    Paper: https://arxiv.org/abs/1706.03762

    More specifically, it uses a encoder-decoder architecture with attention mechanisms embedded inside.
    """

    def __init__(self, vocab_size, hidden_size, num_attention_heads, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(vocab_size=vocab_size, embedding_size=hidden_size)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            sublayer_output_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )

    def forward(self, source, target, source_mask=None, target_mask=None):
        encoder_output = self.encoder(source, mask=source_mask)
        output = self.decoder(
            target,
            encoder_output=encoder_output,
            source_mask=source_mask,
            target_mask=target_mask,
        )
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

        self.positional_encoder = PositionalEncoding(hidden_size=embedding_size)

        self.sublayers = [
            EncoderSublayer(
                hidden_size=embedding_size,
                ff_hidden_size=embedding_size * 4,
            )
            for _ in range(6)
        ]

    def forward(self, x, mask=None):
        x = self.embedding(x) * Tensor(self.embedding_size).sqrt()
        x = self.positional_encoder(x)
        for sublayer in self.sublayers:
            x = sublayer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, sublayer_output_size, num_attention_heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, sublayer_output_size)
        self.positional_encoder = PositionalEncoding(hidden_size=sublayer_output_size)

        self.sublayers = [
            DecoderSublayer(
                hidden_size=sublayer_output_size,
                ff_hidden_size=sublayer_output_size * 4,
                num_attention_heads=num_attention_heads,
            )
            for _ in range(6)
        ]
        self.linear = nn.Linear(sublayer_output_size, output_size=vocab_size)

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        for sublayer in self.sublayers:
            x = sublayer(x, encoder_output, source_mask, target_mask)
        x = self.linear(x)
        output = functional.softmax(x)
        return output


class AddAndNorm(nn.Module):
    def __init__(self, input_size, dropout_prob=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)

        # 5.4 in Paper. Apply Dropout to the output of each layer before
        # adding to sublayer input
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, previous_layer: nn.Module):
        # Residual connection from input x
        return x + self.dropout(previous_layer(self.layer_norm(x)))


class EncoderSublayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        ff_hidden_size=2048,
        dropout_prob=0.1,
        num_attention_heads=2,
    ):
        super().__init__()

        # Multi-head self attention
        self.add_and_norm1 = AddAndNorm(hidden_size)
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads, hidden_size=hidden_size
        )

        # Position-wise feedforward
        self.add_and_norm2 = AddAndNorm(hidden_size)
        self.feedforward = FeedForward(
            fc_input_size=hidden_size,
            hidden_size=ff_hidden_size,
            dropout_prob=dropout_prob,
        )

    def forward(self, x, mask):
        # Multi-head self attention
        x = self.add_and_norm1(
            x, lambda x_: self.multi_head_attention(x_, x_, x_, mask=mask)
        )

        # Position-wise feedforward
        x = self.add_and_norm2(x, self.feedforward)
        return x


class DecoderSublayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        ff_hidden_size=2048,
        dropout_prob=0.1,
        num_attention_heads=2,
    ):
        super().__init__()

        self.add_and_norm1 = AddAndNorm(hidden_size)
        self.masked_multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads, hidden_size=hidden_size
        )

        self.add_and_norm2 = AddAndNorm(hidden_size)
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads, hidden_size=hidden_size
        )

        self.add_and_norm3 = AddAndNorm(hidden_size)
        self.feedforward = FeedForward(
            fc_input_size=hidden_size,
            hidden_size=ff_hidden_size,
            dropout_prob=dropout_prob,
        )

    def forward(self, x, encoder_output: Tensor, source_mask, target_mask):
        # Masked Multi-head Attention
        # Figure 1 in Paper.
        x = self.add_and_norm1(
            x, lambda x_: self.masked_multi_head_attention(x_, x_, x_, mask=target_mask)
        )

        # 3.2.3 in Paper. Encoder-Decoder Attention
        # Queries come from previous decoder layer
        # keys, and values from output of encoder
        x = self.add_and_norm2(
            x,
            lambda x_: self.multi_head_attention(
                query=x_, key=encoder_output, value=encoder_output, mask=source_mask
            ),
        )

        # Final Position-wise Feedforward
        x = self.add_and_norm3(x, self.feedforward)
        return x


class PositionalEncoding(nn.Module):
    """
    Implements the Positional Encoding from Section 3.5 in the paper.
    This allows the model to learn to attent to relative positions even
    without the sequence order information.

    PE(pos, 2i) = sin(pos / 10000^(2i/hidden_size))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/hidden_size))
    Where:
        pos is the position
        i is the dimension
    """

    def __init__(self, hidden_size, max_seq_len=5000, dropout_prob=0.1):
        super().__init__()
        pe = np.zeros((max_seq_len, hidden_size), dtype=np.float32)
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        inverse_freq = 1.0 / 10000 ** (np.arange(0, hidden_size, 2) / hidden_size)
        pe[:, 0::2] = np.sin(position * inverse_freq)
        pe[:, 1::2] = np.cos(position * inverse_freq)

        # Shape (max_seq_len, hidden_size)
        self._parameters["pe"] = Tensor(pe, requires_grad=False)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor):
        """
        Takes (batch_size, seq_len, vocab_size) and returns the same shape

        Args:
            x (Tensor): Embedding representation for the input text
        """
        batch_size, seq_len, input_size = x.shape
        positional_embedding = self._parameters["pe"][:seq_len, :].expand(
            (batch_size, seq_len, input_size)
        )
        x = x + positional_embedding
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, fc_input_size, hidden_size, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_input_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Implements the Scaled Dot-Product Attention in Section 3.2.1 in the paper.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None):
        attention_size = Tensor(key.shape[-1])

        # scaled dot product
        # (batch_size, num_heads, sequence_len, sequence_len)
        att_score = (query @ key.transpose(2, 3)) / attention_size.sqrt()

        # mask (optional)
        if mask is not None:
            # broadcast across heads
            att_score = att_score + (mask * -1e9)
        att_score = functional.softmax(att_score)
        att_score = att_score @ value
        return att_score


class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention in Section 3.2.2 in the paper.
    """

    def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.num_heads = num_heads
        self.attention_size = (
            hidden_size // num_heads
        )  # We assume query, key, value all have the same dimension

        # Project query, key, value using linear layers before passing to attention
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        if mask is not None:
            pass

        # We try to avoid explicitly splitting and combining the heads
        # So we are just using matrix multiplication to paralellize everything
        # Then we are going to reshape the resulting output to the correct
        # dimensions
        # 1. Linear Projections
        # (batch_size, num_heads, seq_len, input_size)
        query = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.attention_size)
            .permute(0, 2, 1, 3)
        )
        key = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.attention_size)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.attention_size)
            .permute(0, 2, 1, 3)
        )

        # 2. Apply Attention
        att_score = self.attention(query, key, value, mask=mask)

        att_score = att_score.permute(
            0, 2, 1, 3
        )  # (batch_size, num_heads , seq_len, input_size)
        # Expect (batch_size, seq_len, hidden_size)
        att_score = att_score.view(batch_size, -1, self.num_heads * self.attention_size)
        assert att_score.shape == (
            batch_size,
            query.shape[2],
            self.num_heads * query.shape[3],
        )

        del query, key, value

        return self.fc(att_score)


def tokens_to_onehot(batch_tokens, word2idx):
    # batch_tokens shape: (batch_size, seq_len)
    # return shape: (batch_size, seq_len, vocab_size)
    batch_size, seq_len = batch_tokens.shape
    out = np.zeros((batch_size, seq_len, len(word2idx)), dtype=np.float32)
    for b in range(batch_size):
        for s in range(seq_len):
            token = batch_tokens[b, s]
            idx = word2idx.get(token, 0)
            out[b, s, idx] = 1.0
    return out


def onehot_to_tokens(onehot_vectors, idx2word):
    # onehot_vectors shape: (batch_size, seq_len, vocab_size)
    # return shape: (batch_size, seq_len)
    batch_size, seq_len, vocab_size = onehot_vectors.shape
    batches = []
    for b in range(batch_size):
        seq = ""
        for s in range(seq_len):
            idx = np.argmax(onehot_vectors[b, s])  # Get the index of the max value
            seq += " " + idx2word.get(
                idx, "<UNK>"
            )  # Convert index to token, using <UNK> for unknown
        batches.append(seq)
    return batches


def token_batch_to_indices(token_batch, vocab):
    X = []
    for batch in token_batch:
        seq = []
        for token in batch:
            seq.append(vocab.get(token, 0))
        X.append(seq)
    return np.array(X)


if __name__ == "__main__":
    NUM_EPOCHS = 1000
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "examples/tinyshakespeare.txt"

    data = load_data(url, filename)[:3000]
    data = text_utils.clean_and_tokenize(data)
    vocab = text_utils.create_vocabulary(data, max_features=10000)
    idx2word = {i: w for i, w in enumerate(vocab)}
    print(data.shape, data[:3], len(vocab))

    # one_hot, indices = text_to_one_hot_and_sparse(data, vocab, max_sequence_length=20)
    # print(Tensor(indices)[:3])
    n = int(len(data) * 0.9)
    train_data, test_data = data[:n], data[n:]

    model = Transformer(vocab_size=len(vocab), hidden_size=512, num_attention_heads=8)
    model.train()

    optimizer = optim.Adam(model.parameters, lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        train_X, train_y = text_utils.create_batches(
            train_data, batch_size=64, seq_len=100
        )

        x = token_batch_to_indices(train_X, vocab)
        y = token_batch_to_indices(train_y, vocab)

        # Create masks
        source_mask = Tensor(
            text_utils.create_padding_mask(x, pad_idx=0), requires_grad=False
        )

        batch_size, seq_len = y.shape
        pad_mask = text_utils.create_padding_mask(y, pad_idx=0)
        causal_mask = text_utils.create_causal_mask(seq_len, batch_size)
        target_mask = Tensor(pad_mask + causal_mask, requires_grad=False)

        optimizer.zero_grad()
        # pred_probs is (batch_size, sequence_len, vocabulary_size)
        pred_prob = model(
            x,
            y,
            source_mask,
            target_mask,
        )
        loss = functional.sparse_cross_entropy(pred_prob, y)
        print(f"Epoch {epoch} | Loss: {loss.detach().data}")

        loss.backward()
        optimizer.step()

        if epoch % max(1, (NUM_EPOCHS // 10)) == 0:
            print("----- Evaluation -----")
            model.eval()
            test_X, test_y = text_utils.create_batches(
                test_data, batch_size=1, seq_len=50
            )
            x = token_batch_to_indices(train_X, vocab)
            y = token_batch_to_indices(train_y, vocab)
            pred_prob = model(x, y)
            pred_tokens = onehot_to_tokens(pred_prob.data, idx2word)
            print(f"{' '.join(train_y[0][:15])=}")
            print(f"{pred_tokens[0]=}")
            model.train()
