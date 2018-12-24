import json_lines
import tensorflow as tf
import gensim
import numpy as np
import jsonlines
import pickle

# Defining constants
batch_size = 100
embedding_dim = 300
class_num = 3
epochs = 10
learning_rate = 10e-3
test_size = 10000
dev_size = 10000

# Defining file names
# train_data_file_name = "./resources/snli_1.0_dev.jsonl"
train_data_file_name = "./resources/all.jsonl"
embedding_file_name = "./resources/temp.bin"

# Word embedding file
word_dict = gensim.models.KeyedVectors.load_word2vec_format(embedding_file_name, binary=True)


# Data set representation
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


# tf Graph Input
x = tf.placeholder(tf.float32, [None, 2 * embedding_dim], name="input")
y = tf.placeholder(tf.int32, [None, class_num], name="label")

# Set model weights
W = tf.Variable(tf.random_normal([2 * embedding_dim, class_num], name="weight"))
b = tf.Variable(tf.random_normal([class_num]), name="bias")

# Prediction
z = tf.matmul(x, W) + b
pred = tf.nn.sigmoid(z)

# Loss and optimizer
loss = tf.losses.softmax_cross_entropy(y, pred)
training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add summary for tensorboard
#tf.summary.scalar("loss", loss)
#merge_summary_op = tf.summary.merge_all()

# Create session
sess = tf.Session()
#writer = tf.summary.FileWriter("./graph", sess.graph)
sess.run(tf.global_variables_initializer())

# Get train file length and batch count
count = 0
data_for_counting = Dataset(train_data_file_name)
for sample in data_for_counting.file:
    if sample.get("gold_label") == '-':
        continue
    count += 1
train_size = count
batch_num = 0
if train_size % train_size == 0:
    batch_num = int(train_size / batch_size)
else:
    batch_num = int(train_size / batch_size) + 1

# Training
for epoch_i in range(epochs):
    data = Dataset(train_data_file_name)
    total_loss = 0
    for i in range(batch_num):
        inputs_p, inputs_h, labels = data.next_batch(batch_size)
        temp = np.concatenate((inputs_p, inputs_h), 1)
        input_batch = temp.reshape(-1, 2 * embedding_dim)
        target_batch = labels.reshape(-1, class_num)
        _, loss_batch = sess.run([training_op, loss], feed_dict={x: input_batch, y: target_batch})
        #writer.add_summary(summary, epoch_i * batch_num + i)
        total_loss += loss_batch

        # Calculate accuracy
        print('Accuracy ', accuracy.eval({x: input_batch, y: target_batch}, session=sess))

    print("loss at epoch ", epoch_i, ": ", total_loss / train_size)

    # if epoch_i % 10 == 0:
    #     saver.save(sess, "./model/trial" + str(epoch_i) + ".ckpt")


# Recording uncertainty with index
uncertainty_list = list()
data = Dataset(train_data_file_name)
for index in range(train_size):
    input_p, input_h, label = data.next_batch(1)
    input_embedding = np.concatenate((input_p, input_h), 1)
    predictions = sess.run(pred, feed_dict={x: input_embedding})
    uncertainty = 1 - np.max(predictions)
    uncertainty_list.append((index, uncertainty))

# Sort according to uncertainty
uncertainty_list.sort(key=lambda t: t[1], reverse=True)

# Divide list
test_list = uncertainty_list[0: test_size]
dev_list = uncertainty_list[test_size: dev_size + test_size]
train_list = uncertainty_list[dev_size + test_size:]

lists = dict()
lists["test_list"] = test_list
lists["dev_list"] = dev_list
lists["train_list"] = train_list

f = open("lists.pkl", "wb")
pickle.dump(lists, f)
f.close()

