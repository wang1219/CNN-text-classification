# -*- coding: utf-8 -*-
#
import jieba
import numpy as np


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
    '汽车': [1, 0],
    '财经': [0, 1]
}


def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    labels_data = read_config(data_file)
    labels = []
    x_text = []
    stopwords = get_stop_words('./data/stopwords')
    for label, contents in labels_data.items():
        for doc in contents:
            labels.append(label)
            doc = jieba.cut(doc, cut_all=False)
            doc = [i.strip() for i in doc if i.replace("\n", "")]

            x_text.append(list(set(doc) - set(stopwords)))

    # Generate labels
    all_labels = [label_dict[label] for label in labels]
    y = np.concatenate([all_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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