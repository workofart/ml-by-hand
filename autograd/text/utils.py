from collections import defaultdict
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


def validate_batches(x, y):
    batch_size, seq_len = x.shape
    for b in range(min(4, batch_size)):
        for seq_idx in range(seq_len):
            print("[X]: ", x[b, : seq_idx + 1])
            print("[y]: ", y[b, seq_idx])


def token_batch_to_indices(token_batch, vocab):
    X = []
    for batch in token_batch:
        seq = []
        for token in batch:
            seq.append(vocab.get(token, vocab["<UNK>"]))
        X.append(seq)
    return np.array(X)
