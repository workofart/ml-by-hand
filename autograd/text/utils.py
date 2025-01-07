from collections import defaultdict
from typing import Union
import numpy as np
import re


def create_vocabulary(
    texts,
    max_features: int,
    custom_tokenizer=None,
    special_tokens=["<PAD>", "<SOS>", "<UNK>"],
):
    """
    Create a vocabulary (word->index) from given texts,
    keeping up to max_features most common words.

    """
    token_freq = defaultdict(int)
    for text in texts:
        if custom_tokenizer is None:
            tokens = text.lower().split()
        else:
            tokens = custom_tokenizer(text)

        for t in tokens:
            token_freq[t] += 1

    for i, st in enumerate(special_tokens):
        token_freq[st] = float("inf") - i

    # Sort by frequency
    sorted_words = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    if max_features is not None:
        sorted_words = sorted_words[:max_features]

    # Create word->index mapping
    vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
    return vocab


def text_to_one_hot_and_sparse(
    texts: list, vocabulary: list, max_sequence_length: int, pad_str="<PAD>"
):
    """
    Convert list of texts into a sequential feature matrix using the vocabulary.
    It will do the padding/truncation based on max_sequence_length, then convert to one-hot encoding
    Shape: (batch_size, sequence_length, vocab_size)

    Args:
        texts (list of str): The input sentences or documents.
        vocabulary (dict): A mapping of word -> index. We'll also add "<PAD>"
                           if it’s not already present.
        max_sequence_length (int): The maximum sequence length for truncation/padding.

    Returns:
        one_hot (np.ndarray): shape (batch_size, max_sequence_length, vocab_size)
        matrix  (np.ndarray): shape (batch_size, max_sequence_length) of integer IDs
    """
    batch_size = len(texts)
    vocab_size = len(vocabulary)
    pad_idx = vocabulary[pad_str]

    # Create an integer marix of shape (batch_size, max_sequence_length)
    # filled with pad_idx initially, then we will overwrite with actual indices later
    matrix = np.full(
        (batch_size, max_sequence_length), fill_value=pad_idx, dtype=np.int32
    )

    for i, text in enumerate(texts):
        # Split text into words and convert to indices
        words = text.lower().split()
        # Truncate or pad sequence to max_sequence_length
        words = words[:max_sequence_length]

        for j, word in enumerate(words):
            if word in vocabulary:
                matrix[i, j] = vocabulary[word]
            else:
                matrix[i, j] = vocabulary.get("<UNK>", pad_idx)

    # Convert to one-hot encoding
    # Shape: (batch_size, sequence_length, vocab_size)
    one_hot = np.zeros((batch_size, max_sequence_length, vocab_size))
    for i in range(batch_size):
        for j in range(max_sequence_length):
            idx_in_vocab = matrix[i, j]
            one_hot[i, j, idx_in_vocab] = 1

    return one_hot, matrix


def create_causal_mask(seq_len, batch_size, lookback=False, mask_diagonal=True):
    """
    Creates a causal mask that prevents positions from attending to future (lookforward)
    or past (lookback) positions. 1.0 => masked.

    Args:
        seq_len (int): Length of the sequence
        batch_size (int): Size of the batch
        lookback (bool): If True, masks "past" (i>j). If False, masks "future" (i<j).
        mask_diagonal (bool): If True, the main diagonal is also masked.

    Returns:
        np.ndarray: shape (batch_size, 1, seq_len, seq_len) with 1.0 in masked positions.
    """
    # We want to produce a matrix M of shape (seq_len, seq_len) where
    # M[i,j] = 1 if it is masked, else 0 if it's allowed.

    if lookback:
        # Mask the lower triangle => i>j => row>column => can't attend to "past"
        # If mask_diagonal=True => includes diagonal => i>=j
        # If mask_diagonal=False => strictly below diagonal => i>j
        k_ = 0 if mask_diagonal else -1
        # np.tril(..., k=0) includes diagonal; np.tril(..., k=-1) excludes diagonal
        mask_2d = np.tril(np.ones((seq_len, seq_len), dtype=np.float32), k=k_)
    else:
        # Mask the upper triangle => i<j => can't attend to "future"
        # If mask_diagonal=True => includes diagonal => i<=j => so we do k=0 in np.triu
        # If mask_diagonal=False => strictly above diagonal => i<j => so we do k=1
        k_ = 0 if mask_diagonal else 1
        mask_2d = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=k_)

    # "mask" means 1.0 in forbidden positions.
    # Add batch dimension: (batch_size, 1, seq_len, seq_len)
    mask_4d = mask_2d[np.newaxis, np.newaxis, :, :]
    mask_4d = np.repeat(mask_4d, batch_size, axis=0)
    return mask_4d


def create_padding_mask(token_indices, pad_idx=0, dims=None):
    """
    Creates a padding mask with configurable output dimensions.

    Args:
        token_indices (np.ndarray): shape (batch_size, seq_len) containing token indices
        pad_idx (int): Integer indicating the padding token index
        dims (tuple or None): Desired shape. If None, (batch_size, 1, 1, seq_len).

    Returns:
        np.ndarray: A mask array where positions of 'pad_idx' are 1.0
    """
    pad_positions = (token_indices == pad_idx).astype(np.float32)

    if dims is None:
        # Default shape for standard attention: (batch_size, 1, 1, seq_len)
        return pad_positions[:, np.newaxis, np.newaxis, :]
    else:
        # We will simply reshape pad_positions to the desired dims
        # assuming the number of elements matches.
        mask = pad_positions.reshape(dims)
        return mask


def clean_and_tokenize(
    text, pattern=r"\w+|[^\w\s]|[\n\s]", lowercase=True
) -> list[str]:
    """
    Naive tokenizer split by words

    Args:
        text (str): The entire input text to be tokenized
        pattern (str): Regular expression pattern used for tokenization.
                      Default splits on words, punctuation and whitespace.
        lowercase (bool): Whether to convert tokens to lowercase. Default True.

    Returns:
        list of tokens (str)
    """
    # Split using provided regex pattern
    tokens = np.array(re.findall(pattern, text))

    # Optionally convert to lowercase
    if lowercase:
        tokens = np.vectorize(str.lower)(tokens)

    # Filter out whitespace tokens
    tokens = tokens[(tokens != " ") & (tokens != "\n")]
    return tokens


def create_batches(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    sampling: str = "random",
    return_dict: bool = False,
) -> Union[tuple[np.ndarray, np.ndarray], dict[str, np.ndarray]]:
    """
    Create training batches for both X and y for autoregressive (next token prediction) purposes.

    Args:
        data (np.ndarray): The tokenized data
        batch_size (int): How many samples are in each batch
        seq_len (int): How many tokens are in one sequence
        sampling (str): Sampling strategy - either "random" or "sequential"
        E.g.
            random = [2, 10]
            X = [
                data[2:7],   # [2, 3, 4, 5, 6]
                data[10:15], # [10, 11, 12, 13, 14]
            ]
            sequential = [4, 9]  # Spaced by seq_len
            X = [
                data[4:9],   # [4, 5, 6, 7, 8]
                data[9:14],  # [9, 10, 11, 12, 13]
            ]
        return_dict (bool): If True, returns dict with "inputs" and "labels" keys.
                          If False, returns (X, y) tuple.

    Returns:
        If return_dict is True:
            dict with keys "inputs" and "labels" containing batched data
        If return_dict is False:
            tuple of (X, y) containing batched training features and labels
    """
    n_samples = len(data)

    if sampling == "random":
        sample_indices = np.random.randint(0, n_samples - seq_len, size=(batch_size,))
    elif sampling == "sequential":
        start_idx = np.random.randint(0, n_samples - batch_size * seq_len)
        sample_indices = np.arange(start_idx, start_idx + batch_size * seq_len, seq_len)
    else:
        raise ValueError("sampling must be either 'random' or 'sequential'")

    X = np.array([data[i : i + seq_len] for i in sample_indices])
    y = np.array([data[i + 1 : i + seq_len + 1] for i in sample_indices])

    if return_dict:
        return {"inputs": X, "labels": y}
    return X, y


def validate_batches(x, y):
    batch_size, seq_len = x.shape
    for b in range(min(4, batch_size)):
        for seq_idx in range(seq_len):
            print("[X]: ", x[b, : seq_idx + 1])
            print("[y]: ", y[b, seq_idx])


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
