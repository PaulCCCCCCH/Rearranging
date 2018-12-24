import json_lines
import tensorflow as tf
import gensim
import numpy as np

# Defining constants
node_number = 32
batch_size = 1
embedding_dim = 300
class_num = 3

# Defining file names
train_data_file_name = "./resources/snli_1.0_dev.jsonl"
f = open(train_data_file_name)
embedding_file_name = "./resources/temp.bin"

# word_dict = gensim.models.KeyedVectors.load_word2vec_format(embedding_file_name, binary=True)
file = json_lines.reader(f)

content = []
for line in file:
    content.append(line)


def read_data(lines: list):
    labels = []
    inputs = []
    for line in lines:
        sentence1 = []
        sentence2 = []
        label_text = line.get("gold_label")
        if label_text == '-':
            continue
        for word in line.get("sentence1").strip(".").split():
            sentence1.append(np.zeros((300, 1))) # Use word_dict[word] instead of np.zeros

        for word in line.get("sentence2").strip(".").split():
            sentence2.append(np.zeros((300, 1)))

        inputs.append([sentence1, sentence2])

        label = np.zeros((3, 1))
        if label_text == 'neutral':
            label[0] = 1
        elif label_text == 'contradiction':
            label[1] = 1
        elif label_text == 'entailment':
            label[2] = 1

        labels.append(label)
    return inputs, labels


inputs, labels = read_data(content)

sentence1 = tf.placeholder(tf.float32, [100, batch_size, embedding_dim])
sentence2 = tf.placeholder(tf.float32, [100, batch_size, embedding_dim])
label = tf.placeholder(tf.float32, [1, batch_size, 3])

state = np.zeros([batch_size, node_number])
rnn_cell = tf.nn.rnn_cell.LSTMCell(node_number)
output, state = tf.nn.dynamic_rnn(rnn_cell, sentence1, state, time_major=True)

output, logits = tf.contrib.layers.fully_connected(output, num_outputs=class_num)

loss = tf.contrib.seq2seq.sequence_loss(logits, label)
optimizer = tf.train.RMSPropOptimizer(1e-2).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
epoches = 10
for epoch_i in range(epoches):
    for i in range(len(inputs)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                                               feed_dict={sentence1: inputs[i][0],
                                                          label: labels[i]})


