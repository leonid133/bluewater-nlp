import os
import numpy as np
from scipy.stats import truncnorm
from tensorflow import gfile


def read_vocab(txt_path):
    with gfile.Open(txt_path) as inp:
        return [l.rstrip() for l in inp]


class VocabBuilder:
    def __init__(self, initial_vocab_path, initial_vectors_path, excludes=()):
        self.vocab = []
        self.vectors = []
        #
        arg_vecs = np.load(initial_vectors_path)
        print("Initial vectors array shape %s, dtype %s" % (arg_vecs.shape, arg_vecs.dtype))
        if len(arg_vecs.shape) != 2:
            raise ValueError("Rank != 2")
        self.vec_dtype = arg_vecs.dtype
        self.vec_dims = arg_vecs.shape[1]
        vec_mean = arg_vecs.mean()
        vec_std = arg_vecs.std()
        self.vec_init_rv = truncnorm(-2, 2, loc=vec_mean, scale=vec_std)
        #
        excludes = set(excludes)
        #
        excluded_num = 0
        read_num = 0
        #
        with open(initial_vocab_path, encoding='utf-8') as vocab_inp:
            for t, v in zip(vocab_inp, arg_vecs):
                t = t.rstrip()
                if t in excludes:
                    excluded_num += 1
                    continue
                self.vocab.append(t)
                self.vectors.append(v)
                read_num += 1
        print("Added %s tokens, skipped %s" % (read_num, excluded_num))

    def _generate_v(self):
        return self.vec_init_rv.rvs(size=self.vec_dims).astype(self.vec_dtype)

    def append(self, t):
        v = self._generate_v()
        self.vocab.append(t)
        self.vectors.append(v)

    def prepend(self, t):
        v = self._generate_v()
        self.vocab.insert(0, t)
        self.vectors.insert(0, v)

    def write(self, out_path_base):
        vocab_out_path = out_path_base + '.vocab.txt'
        vectors_out_path = out_path_base + '.vectors.npy'
        #
        out_arr = np.stack(self.vectors)
        if len(self.vocab) != out_arr.shape[0]:
            raise ValueError("Vocab size %s != Vectors array shape[0] %s" %
                             (len(self.vocab), out_arr.shape))
        os.makedirs(os.path.dirname(vocab_out_path), exist_ok=True)
        with open(vocab_out_path, mode='w', encoding='utf-8') as out:
            for t in self.vocab:
                out.write(t)
                out.write('\n')
        np.save(vectors_out_path, out_arr)
