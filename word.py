# word.py: word prediction with RNN
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 8/11/2018

# B1: import tensorflow and numpy
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B2: array for alphabet
char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']

# B3: assign array index to each alphabet
# ex: 'a': 0, 'b': 1, 'c': 2  ...
num_dic = {n: i for i, n in enumerate(char_arr)}

# B4: training words and constants
seq_data = ['body', 'dial', 'open', 'rank', 'need', 'wise', 
            'item', 'jury', 'path', 'ease']
n_input = n_class = 26
n_stage = 3 

# B5: input encoder
def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0:3]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(26)[input])
        target_batch.append(target)
    return input_batch, target_batch

# B6: global parameters
learning_rate = 0.01
n_hidden = 128
total_epoch = 100

# B7: placeholders and variables
# the input placeolder should be 3-dimensional 
#     to use tensorflow RNN cells
# note that Y, output label, is 1-dimensional
X = tf.placeholder(tf.float32, [None, n_stage, n_input])
Y = tf.placeholder(tf.int32, [None])
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# B8: two RNN cells and their deep RNN network
# we use dropout in cell 1
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

# B9: RNN output re-ordering and trimming in 3 steps
# [batch_size, n_stage, n_hidden] ->
# [n_stage, batch_size, n_hidden] ->
# [batch_size, n_hidden] 
# this cases the last stage output to be used
# model produces 26 floating point values
# prediction uses argmax to find which index model computes
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b
prediction = tf.cast(tf.argmax(model, 1), tf.int32)

# B10: prints model and prediction information
def info():
    sample = ['body']
    input_batch, target_batch = make_batch(sample)
    m_info, p_info = sess.run([model, prediction], feed_dict={X: input_batch, Y: target_batch})

    print('word:', sample)
    print('model:', m_info)
    print('prediction:', p_info)
    print('predicted character', char_arr[p_info[0]])
    print()

# B11: loss function and optimizer
# this new tool better handles 1-dimensional label
# note the difference in size between logits and labels
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# B12: testing the given set of words
def test(words):
    input_batch, target_batch = make_batch(words)
    guess = sess.run(prediction, feed_dict={X: input_batch, Y: target_batch})

    for i, seq in enumerate(words):
        print(seq[0:3], char_arr[guess[i]])
    print()

# B13: session and main flow
sess = tf.Session()
sess.run(tf.global_variables_initializer())

test(seq_data)
input_batch, target_batch = make_batch(seq_data)
for epoch in range(total_epoch):
    _, error = sess.run([optimizer, loss], feed_dict={X: input_batch, Y: target_batch})
    print('epoch: %04d' % epoch, 'error = %.4f' % error)

print()
info()
test(seq_data)
seq_data = ['bod', 'bad', 'boe', 'nar', 'nef', 'zzz']
test(seq_data)

