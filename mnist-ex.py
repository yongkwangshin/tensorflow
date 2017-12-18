from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST", one_hot=True)

# data
images = mnist.test.images

# label
labels = mnist.test.labels

# original data
images = mnist.test.images.reshape([-1, 28, 28])
print(images.shape)
print(images[0])

# label
print(labels.shape)
print(labels[0])

