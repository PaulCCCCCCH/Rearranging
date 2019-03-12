import json_lines
import tensorflow as tf
import gensim
import numpy as np
import jsonlines
import pickle
from random import shuffle


vocab_dict = dict()

# data_file = "./resources/stub.jsonl"
data_file = "./resources/all.jsonl"
data = json_lines.reader(open(data_file))
vocab = set()
vocab_count = 0

for sample in data:
    sentence1 = map(str.lower, sample.get("sentence1").strip(".").split())
    sentence2 = map(str.lower, sample.get("sentence2").strip(".").split())

    for word in sentence1:
        if word not in vocab:
            vocab.add(word)
            vocab_count += 1

    for word in sentence2:
        if word not in vocab:
            vocab.add(word)
            vocab_count += 1

print(len(vocab))


vocab_length = len(vocab)
indices = list(range(vocab_length))
shuffle(indices)
vocab_pairs = list(zip(list(vocab), indices))


for pair in vocab_pairs:
    vocab_dict[pair[0]] = pair[1]


f = open('./vocab_dict.pkl', 'wb')
pickle.dump(vocab_dict, f)
f.close()
