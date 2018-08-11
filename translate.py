# translate.py: english to korean translation with RNN
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 8/11/2018

# B1: import related packages
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B2: 41 characters used in the dictionary
# S, E, and P are special characters used in RNN
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)
n_class = n_input = dic_len

# B3: training data
# the english word should be of length 4, and the korean 2
seq_data = [['word', '단어'], ['wood', '나무'], ['game', '놀이'], ['girl', '소녀'], ['kiss', '키스'], ['love', '사랑']]

# B4: one-hot encoding function
def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    return input_batch, output_batch, target_batch

# B5: global parameters
learning_rate = 0.01
n_hidden = 128
total_epoch = 100

# B6: in Seq2Seq RNN, we use the following placeholder type
# for the encoder and decoder:
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])

# B7: in Seq2Seq RNN, we use the following placeholder type
# for the output: [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])

# B8: encoder cell definition
# we use dropout to avoid overfitting
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    _ , enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

# B9: decoder cell definition
# we use dropout to avoid overfitting
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # in Seq2Seq model, we use the encoder output state
    # as the initial state for the decoder
    dec_outputs, _ = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)

# B10: instead of tf.matmul(outputs, W)+b, 
# we use the 'dense' function in tensorflow layers:
# 'dense' produces [batch_size, time_step, input_size]
# in our case, model shape is (?, ?, 41)
model = tf.layers.dense(dec_outputs, n_class, activation=None)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# B11: print model and target information
def info():
    sample = [['word', '단어']]
    input_batch, output_batch, target_batch = make_batch(sample)
    m_info, t_info = sess.run([model, targets], feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})

    print('input:', sample)
    print('model\n', m_info)
    print('target:', t_info)
    print()

# B12: training session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([opt, cost], feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})
    print('epoch: %04d' % epoch, 'cost: %.4f' % loss)
info()

# B13: test new words
new_data = [['word', 'PP'], ['wodr', 'PP'], ['love', 'PP'], ['loev', 'PP'], ['abcd', 'PP'], ['wide', 'PP'], ['gate', 'PP']]
input_batch, output_batch, target_batch = make_batch(new_data)
prediction = tf.argmax(model, 2)
result = sess.run(prediction, feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})

for i in range(len(new_data)):
    decoded = [char_arr[i] for i in result[i]]
    korean = decoded[0] + decoded[1]
    print('english:', new_data[i][0], ', result:', result[i], ', korean:', korean)

