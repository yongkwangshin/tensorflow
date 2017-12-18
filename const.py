# const.py: constants in tensorflow
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/15/2017

# B1: import tensorflow
import tensorflow as tf

# B2: turn off warning sign
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B3: string constant
hello = tf.constant('hello world!')
print(hello)

# B4: integer constant
a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)  
print(c)

# B5: main session
sess = tf.Session()
print(sess.run(hello))
print(sess.run([a, b, c]))
sess.close()