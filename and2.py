# and2.py: perceptron for AND2
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
bias = 1.

# B4: training data
train_in = [[T, T, bias], [T, F, bias], [F, T, bias], [F, F, bias]]
train_out = [[T], [F], [F], [F]]

# B5: weight matrix definition
w = tf.Variable(tf.random_normal([3, 1]))

# B6: step activation function
# step(x) = { 1 if x > 0; -1 otherwise }
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)

# B7: output and error definitions
output = step(tf.matmul(train_in, w))
error = tf.subtract(train_out, output)
mse = tf.reduce_mean(tf.square(error))

# B8: we do weight update ourselves!
# train_in: 4x3 tensor, error: 4x1 tensor
# so we transpose train_in to obtain delta, 3x1 tensor
# weight change is done with tf.assign
delta = tf.matmul(train_in, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

# B9: tensorflow session 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
err, target = 1, 0
epoch, max_epochs = 0, 100

# B10: print W, b, and sample result
def test():
    print('\nweights/bias\n', sess.run(w))
    print('output\n', sess.run(output))
    print('mse: ', sess.run(mse), '\n')

# B11: main session
test()
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch:', epoch, 'mse:', err)
test()
