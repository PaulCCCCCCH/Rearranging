import json_lines
import tensorflow as tf
import gensim
import numpy as np
import jsonlines
import pickle
from random import shuffle

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

            for word in sample.get("sentence1").strip(".").split():
                if word in word_dict:
                    sentence1 += word_dict[word]
            sentence1 /= 100

            for word in sample.get("sentence2").strip(".").split():
                if word in word_dict:
                    sentence2 += word_dict[word]
            sentence2 /= 100

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
x = tf.placeholder(tf.float32, [None, 2 * embedding_dim])
y = tf.placeholder(tf.int32, [None, class_num])

# Set model weights
W = tf.Variable(tf.random_normal([2 * embedding_dim, class_num], name="weights"))
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

# Saving variables
np.save('weights', sess.run(tf.trainable_variables()[0]))
np.save('bias', sess.run(tf.trainable_variables()[1]))


# Recording uncertainty with index
uncertainty_list_0 = list()
uncertainty_list_1 = list()
uncertainty_list_2 = list()
data = Dataset(train_data_file_name)
for index in range(train_size):
    input_p, input_h, label = data.next_batch(1)
    input_embedding = np.concatenate((input_p, input_h), 1)
    predictions = sess.run(pred, feed_dict={x: input_embedding})
    predictions = predictions.reshape(class_num, 1)
    class_index = label.argmax()
    uncertainty = 1 - predictions[class_index, 0]
    if label[0][0] == 1:
        uncertainty_list_0.append((index, uncertainty))
    elif label[0][1] == 1:
        uncertainty_list_1.append((index, uncertainty))
    elif label[0][2] == 1:
        uncertainty_list_2.append((index, uncertainty))
    else:
        print("The data has no label!")

# Sort according to uncertainty
uncertainty_list_0.sort(key=lambda t: t[1], reverse=True)
uncertainty_list_1.sort(key=lambda t: t[1], reverse=True)
uncertainty_list_2.sort(key=lambda t: t[1], reverse=True)

# Divide list
div_point_1 = test_size // 3
div_point_2 = test_size // 3 * 2

test_list = list(uncertainty_list_0[0: div_point_1])
test_list.extend(uncertainty_list_1[0: div_point_1])
test_list.extend(uncertainty_list_2[0: div_point_1])
shuffle(test_list)

dev_list = list(uncertainty_list_0[div_point_1: div_point_2])
dev_list.extend(uncertainty_list_1[div_point_1: div_point_2])
dev_list.extend(uncertainty_list_2[div_point_1: div_point_2])
shuffle(dev_list)

train_list = list(uncertainty_list_0[div_point_2:])
train_list.extend(uncertainty_list_1[div_point_2:])
train_list.extend(uncertainty_list_2[div_point_2:])
shuffle(train_list)


lists = dict()
lists["test_list"] = test_list
lists["dev_list"] = dev_list
lists["train_list"] = train_list

f = open("lists.pkl", "wb")
pickle.dump(lists, f)
f.close()

