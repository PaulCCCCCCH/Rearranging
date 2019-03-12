import json_lines
import tensorflow as tf
import gensim
import numpy as np
import jsonlines
import pickle
from random import shuffle


# data_file = "./resources/stub.jsonl"
data_file = "./resources/all.jsonl"
data = json_lines.reader(open(data_file))

vocab = dict()

max_len = 0
max_ind = 0
record = None
count = 0
tot = 0

for sample in data:
    sentence1 = sample.get("sentence1").strip(".").split()
    sentence2 = sample.get("sentence2").strip(".").split()
    tot = tot + len(sentence1) + len(sentence2)

    if len(sentence1) > max_len:
        max_len = len(sentence1)
        record = sentence1
    if len(sentence2) > max_len:
        max_len = len(sentence2)
        record = sentence2

    count += 1


print(max_len)
print(record)
print(tot/count)




