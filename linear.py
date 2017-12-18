# linear.py: linear regression 
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/15/2017

# B1: import tensorflow
import tensorflow as tf

# B2: turn off warning sign
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B3: training data
x_tr = [1, 2, 3]
y_tr = [1, 2, 3]

# B4: variables used
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# B5: placeholders used
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# B6: linear function
F = W * X + b

# B7: loss function using mean square error
cost = tf.reduce_mean(tf.square(F - Y))

# B8: how we update weight and bias
opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)

# B9: objective during training
train = opt.minimize(cost)

# B10: session definition
train_tot = 100
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# B11: do training
for i in range(train_tot):
    error, _ = sess.run([cost, train], feed_dict={X: x_tr, Y: y_tr})
    print(i, 'error = %.3f' % error, 'W = %.3f' % sess.run(W), 'b = %.3f' % sess.run(b))

# B12: model testing
test = 5
print('\ntest =', test, 'predict = %.3f' % sess.run(F, feed_dict={X: test}))
