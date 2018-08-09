# softmax.py: MNIST image recognition with softmax
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/17/2017

# B1: import tensorflow
import tensorflow as tf

# B2: turn off warning sign
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B3: import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# B4: model creation
x = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.random_normal([784, 300]))
b1 = tf.Variable(tf.zeros([300]))
y1 = tf.nn.softmax(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.random_normal([300, 10]))
b2 = tf.Variable(tf.zeros([10]))
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

# B5: loss function and optimizer
ans = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(y2), 1))
opt = tf.train.AdamOptimizer().minimize(loss)
#opt = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


# B6: session creation
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# B7: test the first digit in MNIST test set, which is 7.
def test():
    x_train = mnist.test.images[0:1]
    answer = sess.run(y2, feed_dict={x: x_train})
    print('\ny vector is', answer)
    print('my guess is', answer.argmax())
    
train_tot = 10000
batch_size = 100

test()
for i in range(train_tot):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    error, _ = sess.run([loss, opt], feed_dict = \
        {x: batch_xs, ans: batch_ys})
    if i % 100 == 0:
        print("batch", i, "error = %.3f" % error)
test()

correct = tf.equal(tf.argmax(y2, 1), tf.argmax(ans, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
images = mnist.test.images
labels = mnist.test.labels
print("\nmodel accuracy:", sess.run(accuracy, 
    feed_dict={x: images, ans: labels}))
