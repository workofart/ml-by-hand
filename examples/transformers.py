import numpy as np

from autograd.tools.data import load_data
from autograd.text import utils as text_utils
from autograd.tensor import Tensor
from autograd import nn, functional, optim


class Transformer(nn.Module):
    """
    Implements the Paper "Attention is All You Need"
    Paper: https://arxiv.org/abs/1706.03762

    More specifically, it uses a encoder-decoder architecture with attention mechanisms embedded inside.

    Overall Architecture:
    - Encoder-decoder framework (Section 3.1)
    - Encoder: 6 identical layers (Section 3.1)
    - Decoder: 6 identical layers (Section 3.1)
    - Each layer has self-attention or masked self-attention in the decoder (Section 3.2)
      and feed-forward sub-layers
    """

    def __init__(self, vocab_size, hidden_size, num_attention_heads, **kwargs):
        """
        Args:
            vocab_size (int): Vocabulary size for the embeddings
            hidden_size (int): Dimension of model (d_model in the paper)
            num_attention_heads (int): Number of attention heads (Section 3.2.2)
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Encoder and dedcoder each have 6 layers
        self.encoder = Encoder(
            embedding_size=hidden_size, num_attention_heads=num_attention_heads
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_mask: Tensor = None,
        target_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of the transformer

        Args:
            source (Tensor): Source sequence indices (batch_size, seq_len)
            target (Tensor): Target sequence indices (batch_size, seq_len)
            source_mask (Tensor, optional): Source mask to cover padding. Defaults to None.
            target_mask (Tensor, optional): Target mask to cover padding + future tokens (causal mask). Defaults to None.

        Returns:
            Tensor: Probability distribution over entire vocabulary
        """
        encoder_output = self.encoder(
            source, embedding_layer=self.embedding, mask=source_mask
        )
        output = self.decoder(
            target,
            embedding_layer=self.embedding,
            encoder_output=encoder_output,
            source_mask=source_mask,
            target_mask=target_mask,
        )
        return output


class Encoder(nn.Module):
    """
    - Embeddings (Section 3.1) + Positional Encoding (Section 3.5)
    - 6 identical EncoderSublayer (Section 3.1)
    """

    def __init__(self, embedding_size, num_attention_heads):
        super().__init__()
        self.embedding_size = embedding_size

        self.positional_encoder = PositionalEncoding(hidden_size=embedding_size)

        self.sublayers = [
            EncoderSublayer(
                hidden_size=embedding_size,
                ff_hidden_size=embedding_size * 4,
                num_attention_heads=num_attention_heads,
            )
            for _ in range(6)
        ]

    def forward(self, x, embedding_layer, mask=None):
        # Section 3.4 (Embedding Scaling) embedding layer is shared between Encoder and Decoder
        x = embedding_layer(x) * Tensor(self.embedding_size).sqrt()

        # Section 3.5
        x = self.positional_encoder(x)

        # Section 3.1
        for sublayer in self.sublayers:
            x = sublayer(x, mask)
        return x


class Decoder(nn.Module):
    """
    Decoder (Section 3.1)
    - Embedding + PostionalEncoding (Section 3.5)
    - 6 identical DecoderSublayer (Section 3.1)
    - Linear + Softmax to produce probability distribution over entire vocabulary
    """

    def __init__(self, vocab_size, hidden_size, num_attention_heads=2):
        super().__init__()
        self.positional_encoder = PositionalEncoding(hidden_size=hidden_size)

        self.sublayers = [
            DecoderSublayer(
                hidden_size=hidden_size,
                ff_hidden_size=hidden_size * 4,
                num_attention_heads=num_attention_heads,
            )
            for _ in range(6)
        ]
        self.linear = nn.Linear(hidden_size, output_size=vocab_size)

    def forward(self, x, embedding_layer, encoder_output, source_mask, target_mask):
        """
        Args:
            x (Tensor): Target sequence (decoder input)
            embedding_layer (nn.Module): The shared embedding layer between Encoder and Decoder (Section 3.4)
            encoder_output (Tensor): Output of encoder
            source_mask (Tensor): Source (padding) mask
            target_mask (Tensor): Target (causal + padding) mask
        """
        x = embedding_layer(x)

        # Section 3.5
        x = self.positional_encoder(x)

        # Section 3.1
        for sublayer in self.sublayers:
            x = sublayer(x, encoder_output, source_mask, target_mask)

        x = self.linear(x)
        output = functional.softmax(x)
        return output


class ResidualAddAndNorm(nn.Module):
    """
    Implements the residual connection + Layer Normalization
    from Section 3.1 in the paper.
    """

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
    """
    3.1 Encoder Stack in the paper.
    - Multi-head self-attention (Section 3.2.2)
    - Residual connection and layer normalization
    - Position-wise feed-forward (Section 3.3)
    - Residual connection and layer normalization
    """

    def __init__(
        self,
        hidden_size=512,
        ff_hidden_size=2048,
        dropout_prob=0.1,
        num_attention_heads=2,
    ):
        super().__init__()

        # Multi-head self attention
        self.add_and_norm1 = ResidualAddAndNorm(hidden_size)
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads, hidden_size=hidden_size
        )

        # Position-wise feedforward
        self.add_and_norm2 = ResidualAddAndNorm(hidden_size)
        self.feedforward = FeedForward(
            fc_input_size=hidden_size,
            hidden_size=ff_hidden_size,
            dropout_prob=dropout_prob,
        )

    def forward(self, x, mask):
        # (Section 3.2.2) Multi-head self attention
        x = self.add_and_norm1(
            x, lambda x_: self.multi_head_attention(x_, x_, x_, mask=mask)
        )

        # (Section 3.3) Position-wise feedforward
        x = self.add_and_norm2(x, self.feedforward)
        return x


class DecoderSublayer(nn.Module):
    """
    3.1 Decoder Stack in the paper.
    - Masked Multi-head self-attention (Section 3.2.3)
    - Residual connection with layer normalization
    - Encoder-Decoder multi-head attention (Section 3.2.3)
    - Position-wise Feed-forward (Section 3.3)
    """

    def __init__(
        self,
        hidden_size=512,
        ff_hidden_size=2048,
        dropout_prob=0.1,
        num_attention_heads=2,
    ):
        super().__init__()

        # Section 3.2.3 Masked Multi-head self-attention
        self.add_and_norm1 = ResidualAddAndNorm(hidden_size)
        self.masked_multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads, hidden_size=hidden_size
        )

        # Section 3.2.3 Encoder-Decoder Attention in the paper
        self.add_and_norm2 = ResidualAddAndNorm(hidden_size)
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads, hidden_size=hidden_size
        )

        # Section 3.3 Position-wise Feed-forward
        self.add_and_norm3 = ResidualAddAndNorm(hidden_size)
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

        # Section 3.3 Final Position-wise Feedforward
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
    """
    Position-wise Feed-forward Network (Section 3.3)

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

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

    Attention(Q,K,V) = softmax(Q transpose(K) / sqrt(key_dim)) V
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

    Instead of performing a single attention with hidden_size keys, query, and values,
    we project them "num_heads" times with different learned linear projects
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


def inference(model, start_tokens, max_length=50):
    model.eval()
    generated = list(start_tokens)
    for _ in range(max_length):
        seq_len = len(generated)

        # Convert current tokens to indices & mask
        cur_input = text_utils.token_batch_to_indices([generated], vocab)

        source_mask = Tensor(
            text_utils.create_padding_mask(cur_input, pad_idx=0), requires_grad=False
        )
        pad_mask = text_utils.create_padding_mask(cur_input, pad_idx=0)
        causal_mask = text_utils.create_causal_mask(seq_len, 1)
        target_mask = Tensor(pad_mask + causal_mask, requires_grad=False)

        # Create causal + pad mask for this partial sequence
        # Then run model(...) with encoder_output if necessary
        probs = model(cur_input, cur_input, source_mask, target_mask)
        # The last token’s distribution is probs[:, -1, :]
        next_token_id = np.argmax(probs.data[0, -1])
        generated.append(idx2word.get(next_token_id, "<UNK>"))
        # Possibly break if next_token_id is <eos> or similar
    return generated


if __name__ == "__main__":
    NUM_EPOCHS = 30
    seq_len = 80
    batch_size = 16
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "examples/tinyshakespeare.txt"

    data = load_data(url, filename)[:10000]
    data = text_utils.clean_and_tokenize(data)
    vocab = text_utils.create_vocabulary(data, max_features=60000)
    idx2word = {i: w for i, w in enumerate(vocab)}
    sos_idx = vocab["<SOS>"]
    print(data.shape, data[:3], len(vocab))

    # one_hot, indices = text_to_one_hot_and_sparse(data, vocab, max_sequence_length=20)
    # print(Tensor(indices)[:3])
    n = int(len(data) * 0.9)
    train_data, test_data = data[:n], data[n:]

    model = Transformer(vocab_size=len(vocab), hidden_size=512, num_attention_heads=8)
    model.train()

    optimizer = optim.Adam(model.parameters, lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        num_batches = len(train_data) // (batch_size * seq_len)

        for step in range(num_batches):
            train_X, train_y = text_utils.create_batches(
                train_data,
                batch_size=batch_size,
                seq_len=seq_len,
                sampling="sequential",
            )

            x = text_utils.token_batch_to_indices(train_X, vocab)
            y = text_utils.token_batch_to_indices(train_y, vocab)

            # Create masks
            source_mask = Tensor(
                text_utils.create_padding_mask(x, pad_idx=0), requires_grad=False
            )

            # batch_size, seq_len = y.shape
            pad_mask = text_utils.create_padding_mask(y, pad_idx=0)
            causal_mask = text_utils.create_causal_mask(seq_len, batch_size)
            target_mask = Tensor(pad_mask + causal_mask, requires_grad=False)

            optimizer.zero_grad()

            # y has shape (batch_size, seq_len)
            sos_idx = vocab["<SOS>"]

            # Create decoder input by prepending <SOS> and dropping the last token
            y_inp = np.zeros_like(y)
            y_inp[:, 0] = sos_idx
            y_inp[:, 1:] = y[:, :-1]

            # pred_probs is (batch_size, sequence_len, vocabulary_size)
            pred_prob = model(
                x,
                y_inp,  # prepend <SOS> for decoder input
                source_mask,
                target_mask,
            )
            loss = functional.sparse_cross_entropy(pred_prob, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().data
        print(f"Epoch {epoch} | Loss: {total_loss / num_batches}")

        if epoch % max(1, (NUM_EPOCHS // 10)) == 0:
            print("----- Evaluation -----")
            model.eval()
            test_X, test_y = text_utils.create_batches(
                test_data, batch_size=1, seq_len=50, sampling="random"
            )

            x = text_utils.token_batch_to_indices(test_X, vocab)
            y = text_utils.token_batch_to_indices(test_y, vocab)

            pred_tokens = inference(
                model,
                start_tokens=[vocab.get("<SOS>", 0)],
                max_length=50,
            )
            print(f"{' '.join(train_y[0][:15])=}")
            print(f"{pred_tokens}")
            model.train()
