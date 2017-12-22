# tensor.py: variables and placeholders in tensorflow
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/15/2017

# B1: import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# B2: placeholder
x = tf.placeholder(tf.float32, [None, 3])

# B3: variable: we can initialize them using desired values
W = tf.Variable([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
b = tf.Variable([0.2, 0.2])

# B4: or we can initialize them randomly
# random_normal default: mean=0.0 and stddev=1.0
# W = tf.Variable(tf.random_normal([3, 2]))
# b = tf.Variable(tf.random_normal([2, 1]))

# B5: output calculation
output = tf.matmul(x, W) + b

# B6: main session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
input = [[1, 2, 3], [4, 5, 6]]

print("input:", input)
print("W:", sess.run(W))
print("b:", sess.run(b))
print("output:", sess.run(output, feed_dict={x: input}))

