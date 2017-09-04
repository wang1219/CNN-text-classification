# -*- coding: utf-8 -*-
#
import multiprocessing

from gensim.models import word2vec

import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Word2Vec(object):
    def __init__(self, in_put, out_put=None, size=100, window=5, min_count=5, workers=None):
        self.in_put = in_put
        self.out_put = out_put
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers or multiprocessing.cpu_count()

    def setup(self):
        # model = gensim.models.KeyedVectors.load_word2vec_format(self.in_put, binary=True)
        # model.init_sims(replace=True)

        model = word2vec.Word2Vec(
            word2vec.Text8Corpus(self.in_put),
            size=self.size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers)

        # model.save(self.out_put)

        # model1 = word2vec.Word2Vec.load(self.out_put)
        vocab = list(model.wv.vocab.keys())
        print('vocab', vocab)
        try:
            c = model['鼓励']
        except KeyError:
            print("not in vocabulary")
            c = 0
        print('C...............', c)
        # print(model1.similarity('aaa', 'sound'))
        # print('model1', model1['test'])


    def load(self):
        print('self.out_put', self.out_put)
        # return word2vec.Word2Vec.load_word2vec_format(self.out_put)
        return gensim.models.KeyedVectors.load_word2vec_format(self.out_put, binary=True)


if __name__ == '__main__':
    word2Vec = Word2Vec('./data/val_fenci.txt')
    word2Vec.setup()
