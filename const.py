# const.py: constant arithmetic in tensorflow
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/15/2017

# B1: simple add
print(2+3)

# B2: import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# B3: tensorflow add
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
    print(sess.run(x))

# B4: tensor flow multiply
a = tf.constant([2, 2])
b = tf.constant([[0, 1], [2, 3]])
x = tf.add(a, b)
y = tf.multiply(a, b)
with tf.Session() as sess:
    x, y = sess.run([x, y])
    print('x:', x)
    print('y:', y)

# B5: tensor flow matrix multiply
a = tf.constant([[2, 2]])
b = tf.constant([[0, 1], [2, 3]])
z = tf.matmul(a, b)
with tf.Session() as sess:
    z = sess.run(z)
    print('z:', z)

