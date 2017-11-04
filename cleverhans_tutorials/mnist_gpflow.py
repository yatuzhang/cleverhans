import gpflow as GPflow
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def xavier(dims):
    return np.random.randn(*dims)*(2./(dims[-1]+dims[-2]))**0.5

class CNN(GPflow.model.Model):
    def __init__(self, X, Y, b):
        GPflow.model.Model.__init__(self)
        self.X = GPflow.svgp.MinibatchData(X, b, np.random.RandomState(0))
        self.Y = GPflow.svgp.MinibatchData(Y, b, np.random.RandomState(0))

    def forward(self):
        self.conv1 = self.conv_layer(self.X, 1, 32, "conv1")
        self.relu1 = tf.nn.relu(self.conv1)
        self.pool1 = self.max_pool(self.relu1)
        self.conv2 = self.conv_layer(pool1, 32, 64, "conv2")
        self.relu2 = tf.nn.relu(self.conv2)
        self.pool2 = self.max_pool(self.relu2)
        self.fc1 = self.fc_layer(self.pool2, 7*7*64, 1024, "fc1")
        self.relu3 = tf.nn.relu(self.fc1)
        self.fc2 = self.fc_layer(self.relu3, 1024, 10, "fc2")
        self.softmax1 = tf.nn.softmax(self.fc2)
        return self.softmax1

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        filters = GPflow.param.Param(xavier((filter_size, filter_size, in_channels, out_channels)))
        setattr(self, name + "_filters", filters)
        biases = GPflow.param.Param(np.zeros(out_channels))
        setattr(self, name + "_biases", biases)
        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        weights = GPflow.param.Param(xavier((in_size, out_size)))
        setattr(self, name + "_weights", weights)
        biases = GPflow.param.Param(np.zeros(out_size))
        setattr(self, name + "_biases", biases)
        return weights, biases


class LinearMulticlass(GPflow.model.Model):
    def __init__(self, X, Y):
        GPflow.model.Model.__init__(self) # always call the parent constructor
        self.X = X.copy() # X is a numpy array of inputs
        self.Y = Y.copy() # Y is a 1-of-k representation of the labels

        self.num_data, self.input_dim = X.shape
        _, self.num_classes = Y.shape

        #make some parameters
        self.W = GPflow.param.Param(np.random.randn(self.input_dim, self.num_classes))
        self.b = GPflow.param.Param(np.random.randn(self.num_classes))

        # ^^ You must make the parameters attributes of the class for
        # them to be picked up by the model. i.e. this won't work:
        #
        # W = GPflow.param.Param(...    <-- must be self.W

    def build_likelihood(self): # takes no arguments
        p = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b) # Param variables are used as tensorflow arrays.
        return tf.reduce_sum(tf.log(p) * self.Y) # be sure to return a scalar

if __name__ == "__main__":
    X, Y = mnist.train.next_batch(50000)

    cnn = CNN(X, Y, b=100)
    cnn.forward()
    '''
    X = np.vstack([np.random.randn(10,2) + [2,2],
                   np.random.randn(10,2) + [-2,2],
                   np.random.randn(10,2) + [2,-2]])
    Y = np.repeat(np.eye(3), 10, 0)
    m = LinearMulticlass(X, Y)
    m.optimize()
    print(m)
    '''