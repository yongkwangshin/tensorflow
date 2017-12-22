# const.py: constants in tensorflow
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/15/2017

# B1: import tensorflow
import tensorflow as tf

# B2: string constant
hello = tf.constant('hello world!')
print(hello)

# B3: integer constant
a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)  
print(c)

# B4: main session
sess = tf.Session()
print(sess.run(hello))
print(sess.run([a, b, c]))
sess.close()