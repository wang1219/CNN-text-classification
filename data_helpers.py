# -*- coding: utf-8 -*-
#
import jieba
import numpy as np
from gensim.models import word2vec


def get_stop_words(filename):
    result = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            result.append(line.strip())
            line = f.readline()
    return result


def read_config(filename):
    result = {}
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            line = line.split('\t')
            if len(line) != 4:
                line = f.readline()
                continue

            if line[0] not in result:
                result[line[0]] = [line[3]]
            else:
                result[line[0]].append(line[3])

            line = f.readline()
    return result


def write_config(filename, data):
    with open(filename, 'a') as f:
        f.write(data)


label_dict = {
    '汽车': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '财经': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    '科技': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    '时尚': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    '文化': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    '教育': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '娱乐': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    '军事': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    '健康': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    '体育': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}


def load_data_and_labels(data_file, stopwords_file):
    # Load data from files
    labels_data = read_config(data_file)
    labels = []
    x_text = []
    stopwords = get_stop_words(stopwords_file)
    for label, contents in labels_data.items():
        for doc in contents:
            labels.append(label)
            doc = jieba.cut(doc, cut_all=False)
            doc = [i.replace("\n", "").strip() for i in doc if i.replace("\n", "").strip()]

            x_text.append(list(set(doc) - set(stopwords)))

    # Generate labels
    all_labels = [label_dict[label] for label in labels]
    y = np.concatenate([all_labels], 0)
    return [x_text, y]


def word_to_vectors(x_text, size, window=5, min_count=1):
    model = word2vec.Word2Vec(x_text, size=size, window=window, min_count=min_count)
    vocab = list(model.wv.vocab.keys())
    word_vectors = {}
    for word in vocab:
        try:
            word_vectors[word] = model[word]
        except:
            word_vectors[word] = np.random.uniform(-1.0, 1.0, size).astype(np.float32)
    return word_vectors


def get_W(word_vectors, vocab_ids_map, k=100):
    W = np.random.uniform(-1.0, 1.0, size=[len(word_vectors) + 1, k]).astype(np.float32)
    for word, vector in word_vectors.items():
        try:
            word_id = vocab_ids_map[word]
        except KeyError:
            pass
        else:
            W[word_id] = vector
    return W


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
