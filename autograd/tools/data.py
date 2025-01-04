import numpy as np
from collections import defaultdict


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    num_samples = len(X)
    num_test = int(num_samples * test_size)
    indices = np.random.permutation(num_samples)
    X_train, X_test = X[indices[num_test:]], X[indices[:num_test]]
    y_train, y_test = y[indices[num_test:]], y[indices[:num_test]]
    return X_train, X_test, y_train, y_test


def create_vocabulary(texts, max_features: int):
    """
    Create a vocabulary (word->index) from given texts,
    keeping up to max_features most common words.
    """
    word_freq = defaultdict(int)
    for text in texts:
        for word in text.lower().split():
            word_freq[word] += 1

    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    if max_features is not None:
        sorted_words = sorted_words[:max_features]

    # Create word->index mapping
    vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
    return vocab


def text_to_one_hot_and_sparse(texts: list, vocabulary: list, max_sequence_length: int):
    """
    Convert list of texts into a sequential feature matrix using the vocabulary.
    Shape: (batch_size, sequence_length, vocab_size)
    """
    # Initialize matrix with zeros
    matrix = np.zeros((len(texts), max_sequence_length), dtype=np.int32)

    for i, text in enumerate(texts):
        # Split text into words and convert to indices
        words = text.lower().split()
        # Truncate or pad sequence to max_sequence_length
        words = words[:max_sequence_length]  # Truncate if too long

        for j, word in enumerate(words):
            if word in vocabulary:
                matrix[i, j] = vocabulary[word]

    # Convert to one-hot encoding
    # Shape: (batch_size, sequence_length, vocab_size)
    one_hot = np.zeros((len(texts), max_sequence_length, len(vocabulary)))
    for i in range(len(texts)):
        for j in range(max_sequence_length):
            if matrix[i, j] > 0:
                one_hot[i, j, matrix[i, j]] = 1

    return one_hot, matrix
