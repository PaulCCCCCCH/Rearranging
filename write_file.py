import pickle
import jsonlines
import numpy as np
import json_lines
import gensim

train_data_file_name = "./resources/all.jsonl"
embedding_file_name = "./resources/temp.bin"
word_dict = gensim.models.KeyedVectors.load_word2vec_format(embedding_file_name, binary=True)
embedding_dim = 300


class Dataset:
    def __init__(self, file_name):
        self.file = json_lines.reader(open(file_name))

    def next_batch(self, batch_size):
        file = self.file
        labels = []
        inputs_p = []
        inputs_h = []
        cur = 0
        for sample in file:

            sentence1 = np.zeros(embedding_dim)
            sentence2 = np.zeros(embedding_dim)
            label_text = sample.get("gold_label")
            if label_text == '-':
                continue

            count = 0
            for word in sample.get("sentence1").strip(".").split():
                if word in word_dict:
                    sentence1 += word_dict[word]
                    count += 1
                if not count == 0:
                    sentence1 /= count

            count = 0
            for word in sample.get("sentence2").strip(".").split():
                if word in word_dict:
                    sentence2 += word_dict[word]
                    count += 1
                if not count == 0:
                    sentence2 /= count

            inputs_p.append(sentence1)
            inputs_h.append(sentence2)

            label = np.zeros((3, 1))
            if label_text == 'neutral':
                label[0] = 1
            elif label_text == 'contradiction':
                label[1] = 1
            elif label_text == 'entailment':
                label[2] = 1

            labels.append(label)
            cur += 1
            if cur >= batch_size:
                break

        return np.asarray(inputs_p), np.asarray(inputs_h), np.asarray(labels)


def write_file(target_file, index_set, data):
    writer = jsonlines.Writer(target_file)
    index = 0
    for sample in data.file:
        if sample.get("gold_label") == '-':
            continue
        if index in index_set:
            writer.write(sample)
            index_set.remove(index)
        index += 1
        if not index_set:
            break
    writer.close()


def get_index_set(lst):
    s = set()
    for e in lst:
        s.add(e[0])
    return s


f = open("./lists.pkl", "rb")
lists = pickle.load(f)
f.close()

test_list = lists.get("test_list")
train_list = lists.get("train_list")
dev_list = lists.get("dev_list")

train_index_set = get_index_set(train_list)
test_index_set = get_index_set(test_list)
dev_index_set = get_index_set(dev_list)

f_test = open("snli_1.0_test.jsonl", "w")
data = Dataset(train_data_file_name)
write_file(f_test, test_index_set, data)
f_test.close()

f_dev = open("snli_1.0_dev.jsonl", "w")
data = Dataset(train_data_file_name)
write_file(f_dev, dev_index_set, data)
f_dev.close()

f_train = open("snli_1.0_train.jsonl", "w")
data = Dataset(train_data_file_name)
write_file(f_train, train_index_set, data)
f_train.close()

