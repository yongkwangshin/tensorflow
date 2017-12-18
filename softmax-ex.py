# softmax-ex.py: illustration of softmax regression
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/17/2017

import numpy as np
import tensorflow as tf
sess = tf.Session()

j = [0.03, 0.03, 0.01, 0.9, 0.01, 0.01, 0.0025,0.0025, 0.0025, 0.0025]
k = [0,0,0,1,0,0,0,0,0,0]

log = -(tf.log(j))
print(sess.run(log))

prod = k * log
print(sess.run(prod))

i = tf.reduce_sum(prod)
print(sess.run(i))

