# and2.py: perceptron for XOR2
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/13/2017

# B1: import tensorflow
import tensorflow as tf

# B2: turn off warning sign
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B3: constants
T, F = 1., -1.

# B4: training data
train_in = [[T, T],  [T, F],  [F, T],  [F, F]]
train_out = [[F], [T], [T], [F]]

# B4: hidden layer 1 definition
w1 = tf.Variable(tf.random_normal([2, 2], seed=123))
b1 = tf.Variable(tf.zeros([2]))
out1 = tf.tanh(tf.add(tf.matmul(train_in, w1), b1))

# hidden layer 2 definition
w3 = tf.Variable(tf.random_normal([2, 2], seed=123))
b3 = tf.Variable(tf.zeros([2]))
out3 = tf.tanh(tf.add(tf.matmul(out1, w3), b3))

# B5: output layer definition
w2 = tf.Variable(tf.random_normal([2, 1], seed=123))
b2 = tf.Variable(tf.zeros([1]))
out2 = tf.tanh(tf.add(tf.matmul(out3, w2), b2))

# B6: error calculation
error = tf.subtract(train_out, out2)
mse = tf.reduce_mean(tf.square(error))

# B7: training objective
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

# B8: tensorflow session preparation
sess = tf.Session()
sess.run(tf.global_variables_initializer())
err, target = 1, 0.01
epoch, max_epochs = 0, 1000

# B9: print W, b, and sample result
def test():
    print('\nweight 1\n', sess.run(w1))
    print('bias 1\n', sess.run(b1))
    print('\nweight 3\n', sess.run(w3))
    print('bias 3\n', sess.run(b3))
    print('weight 2\n', sess.run(w2))
    print('bias 2\n', sess.run(b2))
    print('output\n', sess.run(out2))
    print('mse: ', sess.run(mse), '\n')

# B10: main session
test()
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch:', epoch, ', mse:', err)
test()
