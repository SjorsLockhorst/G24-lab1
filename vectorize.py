from collections import Counter

import numpy as np

from extract import read_data


def vectorize(sentence, vocabulary):
    vec = np.zeros(len(vocabulary))
    words = sentence.split()
    counts = Counter(words)
    for word, freq in counts.items():
        if word in vocabulary:
            vec[vocabulary[word]] = freq
    return vec


def create_bag_of_words(x):
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


if __name__ == "__main__":
    x, y = read_data()
    training_data, vocabulary = create_bag_of_words(x)
    print(vectorize("thank you", vocabulary))
