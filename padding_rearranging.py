import json_lines
import tensorflow as tf
import gensim
import numpy as np

# Defining constants
node_number = 300
batch_size = 3
embedding_dim = 300
class_num = 3
max_len = 50
epochs = 50
train_size = 9000

# Defining file names
train_data_file_name = "./resources/snli_1.0_dev.jsonl"
embedding_file_name = "./resources/temp.bin"

word_dict = gensim.models.KeyedVectors.load_word2vec_format(embedding_file_name, binary=True)


class Dataset:
    def __init__(self, file_name):
        self.file = json_lines.reader(open(file_name))

    def next_batch(self, batch_size):
        file = self.file
        labels = []
        inputs_p = []
        inputs_h = []
        count = 0
        for sample in file:

            sentence1 = np.zeros((max_len, embedding_dim))
            sentence2 = np.zeros((max_len, embedding_dim))
            label_text = sample.get("gold_label")
            if label_text == '-':
                continue
            cur = 0
            for word in sample.get("sentence1").strip(".").split():
                if word in word_dict:
                    sentence1[cur] = word_dict[word]
                cur += 1

            cur = 0
            for word in sample.get("sentence2").strip(".").split():
                if word in word_dict:
                    sentence2[cur] = word_dict[word]
                cur += 1

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
            count += 1
            if count >= batch_size:
                break
        return np.asarray(inputs_p), np.asarray(inputs_h), np.asarray(labels)


x = tf.placeholder(tf.float32, [batch_size, 2 * embedding_dim * max_len])
y = tf.placeholder(tf.float32, [batch_size, class_num])
z = tf.layers.dense(x, units=class_num)
# outputs = tf.nn.sigmoid(z)
loss = tf.losses.sigmoid_cross_entropy(y, z)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=10e-3)
training_op = optimizer.minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    batch_num = int(train_size / batch_size)
    for epoch_i in range(epochs):
        data = Dataset(train_data_file_name)
        total_loss = 0
        for i in range(batch_num):

            inputs_p, inputs_h, labels = data.next_batch(batch_size)
            temp = np.concatenate((inputs_p, inputs_h), 1)
            input_batch = temp.reshape(batch_size, 2 * embedding_dim * max_len)
            target_batch = labels.reshape(batch_size, class_num)
            _, loss_batch = sess.run([training_op, loss], feed_dict={x: input_batch, y: target_batch})
            total_loss += loss_batch
        print(total_loss / batch_size)


