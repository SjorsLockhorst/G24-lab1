from collections import Counter

import numpy as np

from extract import read_data


def vectorize(sentence, vocabulary):
    """Vectorize a sentence based on vocabulary."""

    # Create vector
    vec = np.zeros(len(vocabulary))

    # Split into seperate words
    words = sentence.lower().split()
    counts = Counter(words)
    for word, freq in counts.items():
        if word in vocabulary:
            vec[vocabulary[word]] = freq
    return vec


def vectorize_all(sentences, vocabulary):
    """Vectorize some amount of sentences based on vocabulary."""
    # TODO: Make sparse matrix
    matrix = np.zeros((len(sentences), len(vocabulary)))

    for sent_id, sent in enumerate(sentences):
        vec = vectorize(sent, vocabulary)
        matrix[
            sent_id:,
        ] = vec
    return matrix


def create_bag_of_words(x):
    """Create a matrix bag of words representation of data x."""
    word_sentence_map = {}
    for sent_id, sentence in enumerate(x):
        words = sentence.split()
        counts = Counter(words)
        for word, freq in counts.items():
            if word not in word_sentence_map:
                word_sentence_map[word] = {sent_id: freq}
            else:
                word_sentence_map[word][sent_id] = freq

    feature_matrix = np.zeros((len(word_sentence_map), len(x)))
    for word_id, word in enumerate(word_sentence_map):
        sentences = word_sentence_map[word]
        for sent_id, freq in sentences.items():
            feature_matrix[word_id][sent_id] = freq
    return feature_matrix, {word: idx for idx, word in enumerate(word_sentence_map)}
