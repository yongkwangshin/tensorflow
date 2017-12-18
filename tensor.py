# tensor.py: variables and placeholders in tensorflow
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/15/2017

# B1: import tensorflow
import tensorflow as tf

# B2: turn off warning sign
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B3: placeholder and variable
X = tf.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable([0.5, -0.5])

# B4: output calculation
output = tf.matmul(X, W) + b

# B5: main session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
input = [[1, 2, 3], [4, 5, 6]]

print("input:", input, '\n')
print("W:", sess.run(W), '\n')
print("b:", sess.run(b), '\n')
print("output:", sess.run(output, feed_dict={X: input}))
sess.close()
