import logging
import os
import re
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np
from tqdm import tqdm

from autograd import nn
from autograd.functional import Softmax
from autograd.text.tokenizer import BytePairEncoder

logger = logging.getLogger(__name__)


def create_vocabulary(
    texts: List[str],
    max_features: Optional[int],
    custom_tokenizer: Optional[Callable[[str], List[str]]] = None,
    special_tokens: List[str] = ["<PAD>", "<SOS>", "<UNK>"],
) -> Dict[str, int]:
    """
    Create a vocabulary (word->index) from given texts,
    keeping up to max_features most common words.

    Examples:
        >>> texts = ["Hello world", "Hello there", "World peace"]
        >>> vocab = create_vocabulary(texts, max_features=5)
        >>> print(vocab)
        {'hello': 0, 'world': 1, 'there': 2, 'peace': 3}  # Order and exact indices may vary.
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
    texts: List[str],
    vocabulary: Dict[str, int],
    max_sequence_length: int,
    pad_str: str = "<PAD>",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert list of texts into a sequential feature matrix using the vocabulary.
    It will do the padding/truncation based on max_sequence_length, then convert to one-hot encoding
    Shape: (batch_size, sequence_length, vocab_size)

    Args:
        texts (list of str): The input sentences or documents.
        vocabulary (dict): A mapping of word -> index. We'll also add "<PAD>"
                           if it’s not already present.
        max_sequence_length (int): The maximum sequence length for truncation/padding.
        pad_str (str): The padding string.

    Returns:
        one_hot (np.ndarray): shape (batch_size, max_sequence_length, vocab_size)
        matrix  (np.ndarray): shape (batch_size, max_sequence_length) of integer IDs

    Examples:
        >>> texts = ["Hello world", "Hello there"]
        >>> vocab = {"hello": 0, "world": 1, "there": 2, "<PAD>": 3, "<UNK>": 4}
        >>> one_hot, matrix = text_to_one_hot_and_sparse(texts, vocabulary=vocab, max_sequence_length=4)
        >>> print(matrix)
        [[0, 1, 3, 3],
         [0, 2, 3, 3]]
        >>> print(one_hot.shape)
        (2, 4, 5)
    """
    batch_size = len(texts)
    vocab_size = len(vocabulary)
    pad_idx = vocabulary[pad_str]

    # Create an integer matrix of shape (batch_size, max_sequence_length)
    # filled with pad_idx initially, then we overwrite with actual indices later
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


def create_causal_mask(
    seq_len: int,
    batch_size: int,
    lookback: bool = False,
    mask_diagonal: bool = False,
) -> np.ndarray:
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

    Examples:
        >>> mask = create_causal_mask(seq_len=5, batch_size=2)
        >>> print(mask.shape)
        (2, 1, 5, 5)
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


def create_padding_mask(
    token_indices: np.ndarray,
    pad_idx: int = 0,
    dims: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Creates a padding mask with configurable output dimensions.

    Args:
        token_indices (np.ndarray): shape (batch_size, seq_len) containing token indices
        pad_idx (int): Integer indicating the padding token index
        dims (tuple or None): Desired shape. If None, (batch_size, 1, 1, seq_len).

    Returns:
        np.ndarray: A mask array where positions of 'pad_idx' are 1.0

    Examples:
        >>> token_indices = np.array([[1, 0, 2], [0, 3, 4]])
        >>> mask = create_padding_mask(token_indices, pad_idx=0)
        >>> print(mask.shape)
        (2, 1, 1, 3)
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
    text: str, pattern: str = r"\w+|[^\w\s]|[\n\s]", lowercase: bool = True
) -> List[str]:
    """
    Naive tokenizer split by words

    Args:
        text (str): The entire input text to be tokenized
        pattern (str): Regular expression pattern used for tokenization.
                       Default splits on words, punctuation and whitespace.
        lowercase (bool): Whether to convert tokens to lowercase. Default True.

    Returns:
        list of tokens (str)

    Examples:
        >>> text = "Hello, world! \nNew line."
        >>> tokens = clean_and_tokenize(text)
        >>> print(tokens)
        ['hello', ',', 'world', '!', 'new', 'line', '.']
    """
    tokens = np.array(re.findall(pattern, text))

    if lowercase:
        tokens = np.vectorize(str.lower)(tokens)

    # Filter out whitespace tokens
    tokens = tokens[(tokens != " ") & (tokens != "\n")]
    return tokens.tolist()


def validate_batches(x: np.ndarray, y: np.ndarray) -> None:
    batch_size, seq_len = x.shape
    for b in range(min(4, batch_size)):
        for seq_idx in range(seq_len):
            print("[X]: ", x[b, : seq_idx + 1])
            print("[y]: ", y[b, seq_idx])


def token_batch_to_indices(
    token_batch: List[List[str]],
    vocab: Dict[Union[str, bytes], int],
) -> np.ndarray:
    """
    Convert a batch of token lists to a matrix of token indices using a given vocabulary.

    Args:
        token_batch (List[List[str]]): A list of tokenized sentences (each a list of strings).
        vocab (Dict[Union[str, bytes], int]): A vocabulary mapping tokens to integer indices.

    Returns:
        np.ndarray: A matrix of shape (batch_size, sequence_length) containing token indices.

    Examples:
        >>> token_batch = [["hello", "world"], ["this", "test"]]
        >>> vocab = {"hello": 0, "world": 1, "this": 2, "test": 3, b"<UNK>": 4}
        >>> indices = token_batch_to_indices(token_batch, vocab)
        >>> print(indices)
        [[0, 1],
         [2, 3]]
    """
    X: List[List[int]] = []
    for batch in token_batch:
        seq: List[int] = []
        for token in batch:
            seq.append(vocab.get(token, vocab[b"<UNK>"]))
        X.append(seq)
    return np.array(X, dtype=np.int32)


def inference(
    model: nn.Module,
    prediction_func: nn.AbstractLLMForwardFn,
    bpe: BytePairEncoder,
    start_tokens: Optional[str] = None,
    groundtruth_data: Optional[np.ndarray] = None,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    """
    Perform model inference in one of two modes:

    - Auto-regressive inference (normal) if `groundtruth_data` is None.
        We will continuously feed the model's generated tokens back to the model
        to generate the next token (i.e. next token is conditioned on all previously
        generated tokens).
    - Teacher forcing inference if `groundtruth_data` is provided.
        - In teacher forcing mode, temperature is overridden to 0.0 and top_k to 1 to yield argmax behavior.
        - Generates text by teacher forcing on `groundtruth_data`. At each time step, we feed the model the ground truth tokens up to that point, and measure or collect the predicted next token.

    Args:
        model (nn.Module): The model to run inference on.
        prediction_func (nn.AbstractLLMForwardFn): The forward function that implements the AbstractLLMForwardFn interface.
        bpe (BytePairEncoder): BPE tokenizer.
        start_tokens (Optional[str]): The initial token string (e.g. "<SOS>").
        groundtruth_data (Optional[np.ndarray]): If provided, teacher forcing mode is used.
        max_length (int): The maximum length of tokens to run.
        temperature (float): The amount of randomness in sampling
            > 1.0 => more random
            < 1.0 => less random.
        top_k (Optional[int]): If set, only keep the top_k tokens from the distribution.

    Returns:
        str: The generated text after inference.

    Examples:
        >>> # Dummy implementations for demonstration:
        >>> class DummyModel(nn.Module):
        ...     def forward(self, x): return x
        >>> class DummyLLMForward(nn.AbstractLLMForwardFn):
        ...     def sample(self, model, batch_data): return (np.array([[0,1,2]]), None)
        ...     def train(self, model, batch_data): return (np.array([[0,1,2]]), None)
        >>> model = DummyModel()
        >>> forward_fn = DummyLLMForward()
        >>> bpe = BytePairEncoder(num_merges=10)
        >>> # Auto-regressive mode:
        >>> prediction_text = inference(model, forward_fn, bpe, start_tokens="<SOS>", max_length=10)
        >>> print(prediction_text)
    """

    def sample_next_token(logits: np.ndarray, temp: float, k: Optional[int]) -> int:
        # If temp <= 0 (teacher forcing), use argmax.
        if temp <= 0:
            return int(np.argmax(logits))
        # Otherwise, apply temperature scaling and top-k filtering.
        scaled_logits = logits / temp
        if k is not None and k < len(scaled_logits):
            top_indices = np.argpartition(scaled_logits, -k)[-k:]
            filtered_logits = np.full_like(scaled_logits, -np.inf)
            filtered_logits[top_indices] = scaled_logits[top_indices]
            scaled_logits = filtered_logits
        probabilities = Softmax().forward(scaled_logits)
        return int(np.random.choice(len(probabilities), size=1, p=probabilities))

    # Determine mode and set up initial values.
    teacher_forcing = groundtruth_data is not None
    if teacher_forcing:
        temperature, top_k = 0.0, 1
        # We only run for as many steps as there are ground-truth tokens minus one.
        num_steps = max(0, min(max_length, len(groundtruth_data)) - 1)
        output_ids = [int(groundtruth_data[0])]
    else:
        start_tokens = start_tokens or "<SOS>"
        output_ids = list(bpe.encode(start_tokens))
        num_steps = max_length - len(output_ids)

    # Main loop: decide input tokens based on the mode.
    # for i in tqdm(range(num_steps), desc="Inference", leave=False):
    for i in range(num_steps):
        current_input = groundtruth_data[: i + 1] if teacher_forcing else output_ids
        logits = prediction_func(
            model=model, batch_data=np.array([current_input]), mode="sample"
        )[0].data[0, -1]
        output_ids.append(sample_next_token(logits, temperature, top_k))
        print(bpe.decode([sample_next_token(logits, temperature, top_k)]), end="", flush=True)
    
    print("\n")

    if teacher_forcing:
        groundtruth_text = "\n".join(
            bpe.decode(groundtruth_data.tolist()).split("<|endoftext|>")
        )
        logger.info(f"Teacher forcing mode on!!\nGroundtruth:\n{groundtruth_text}")

    # prediction_text = "\n\n".join(bpe.decode(output_ids).split("<|endoftext|>"))
    # logger.info(f"Prediction:\n\n{prediction_text}")

    # return prediction_text


def load_wiki_simple() -> str:
    from autograd.tools.data import load_data

    if not os.path.exists("training_data/wiki_simple_english.txt"):
        print("Downloading data...")
        os.system(
            "curl -L -o examples/plain-text-wikipedia-simpleenglish.zip https://www.kaggle.com/api/v1/datasets/download/ffatty/plain-text-wikipedia-simpleenglish"
        )
        os.system("unzip examples/plain-text-wikipedia-simpleenglish.zip -d examples")
        os.system("rm -rf examples/1of2")
        os.system("rm -rf examples/2of2")
        os.system("mv examples/AllCombined.txt training_data/wiki_simple_english.txt")

    data = load_data(
        "training_data/wiki_simple_english.txt",
        "training_data/wiki_simple_english.txt",
    )
    logger.info(f"{len(data)} characters in the entire dataset. Sample: \n{data[:100]}")
    return data


def load_shakespeare_mini() -> str:
    from autograd.tools.data import load_data

    data = load_data(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "training_data/tinyshakespeare.txt",
    )
    logger.info(f"{len(data)} characters in the entire dataset. Sample: \n{data[:100]}")
    return data
