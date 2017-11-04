from tensorflow.examples.tutorials.mnist import input_data
import gpflow as GPflow
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X, Y = mnist.train.next_batch(50000)
Xs, Ys = mnist.validation.next_batch(5000)

rand = lambda s: np.reshape(np.random.randn(np.prod(s)), s)

def xavier(dims):
    return np.random.randn(*dims)*(2./(dims[-1]+dims[-2]))**0.5

class CNN(GPflow.model.Model):
    def __init__(self, X, Y, b):
        GPflow.model.Model.__init__(self)
        self.filter_size = 3
        self.X = GPflow.svgp.MinibatchData(np.reshape(X, (-1, 28, 28, 1)), b, np.random.RandomState(0))
        self.Y = GPflow.svgp.MinibatchData(Y, b, np.random.RandomState(0))
        self.conv1_filters = GPflow.param.Param(xavier((self.filter_size, self.filter_size, 1, 32)))
        self.conv1_biases = GPflow.param.Param(np.zeros(32))
        self.conv2_filters = GPflow.param.Param(xavier((self.filter_size, self.filter_size, 32, 64)))
        self.conv2_biases = GPflow.param.Param(np.zeros(64))
        self.fc1_weights = GPflow.param.Param(xavier((7*7*64, 1024)))
        self.fc1_biases = GPflow.param.Param(np.zeros(1024))
        self.fc2_weights = GPflow.param.Param(xavier((1024, 10)))
        self.fc2_biases = GPflow.param.Param(np.zeros(10))

    def build_predict(self, Xnew):
        self.conv1 = self.conv_layer(self.X, self.conv1_filters, self.conv1_biases, "conv1")
        self.relu1 = tf.nn.relu(self.conv1)
        self.pool1 = self.max_pool(self.relu1, "pool1")
        self.conv2 = self.conv_layer(self.pool1, self.conv2_filters, self.conv2_biases, "conv2")
        self.relu2 = tf.nn.relu(self.conv2)
        self.pool2 = self.max_pool(self.relu2, "pool2")
        self.fc1 = self.fc_layer(self.pool2, self.fc1_weights, self.fc1_biases, 7*7*64, "fc1")
        self.relu3 = tf.nn.relu(self.fc1)
        self.fc2 = self.fc_layer(self.relu3, self.fc2_weights, self.fc2_biases, 1024, "fc2")
        self.softmax1 = tf.nn.softmax(self.fc2)
        return self.softmax1

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, filt, bias, name):
        with tf.variable_scope(name):
            conv = tf.nn.conv2d(tf.to_float(bottom), tf.to_float(filt), [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, tf.to_float(bias))
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, weights, bias, in_size, name):
        with tf.variable_scope(name):
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, tf.to_float(weights)), tf.to_float(bias))

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        filter_name = name + "_filters"
        bias_name = name + "_biases"
        filters = GPflow.param.Param(xavier((filter_size, filter_size, in_channels, out_channels)))
        setattr(self, filter_name, filters)
        biases = GPflow.param.Param(np.zeros(out_channels))
        setattr(self, bias_name, biases)
        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        weights = GPflow.param.Param(xavier((in_size, out_size)))
        setattr(self, name + "_weights", weights)
        biases = GPflow.param.Param(np.zeros(out_size))
        setattr(self, name + "_biases", biases)
        return getattr(self, name + "_weights"), getattr(self, name + "_biases")

    def build_likelihood(self):
        Y_preds = self.build_predict(tf.reshape(self.X, [-1, 28, 28, 1]))
        return tf.reduce_mean(tf.reduce_sum(tf.to_float(self.Y) * tf.to_float(tf.log(tf.to_float(Y_preds))), 1))
        
    @GPflow.param.AutoFlow((tf.float32, [None, None]), (tf.float32, [None, None]))
    def accuracy(self, Xnew, Ynew):
        Y_preds = tf.argmax(self.build_predict(tf.reshape(self.X, [-1, 28, 28, 1])), 1)
        Y_true = tf.argmax(Ynew, 1)
        correct = tf.cast(tf.equal(Y_preds, Y_true), tf.float32)
        return tf.reduce_mean(correct)

class Linear_model(GPflow.model.Model):
    def __init__(self, X, Y, b):
        GPflow.model.Model.__init__(self)
        self.X = GPflow.svgp.MinibatchData(X, b, np.random.RandomState(0))
        self.Y = GPflow.svgp.MinibatchData(Y, b, np.random.RandomState(0))
        self.W = GPflow.param.Param(rand((X.shape[1], Y.shape[1])))
        self.b = GPflow.param.Param(np.zeros(Y.shape[1]))
            
    def build_likelihood(self):
        Y_preds = self.build_predict(self.X)
        return tf.reduce_mean(tf.reduce_sum(tf.to_float(self.Y) * tf.to_float(tf.log(tf.to_float(Y_preds))), 1))
        
    def build_predict(self, Xnew):
        return tf.nn.softmax(tf.matmul(Xnew, self.W) + self.b)
        
    @GPflow.param.AutoFlow((tf.float32, [None, None]), (tf.float32, [None, None]))
    def accuracy(self, Xnew, Ynew):
        Y_preds = tf.argmax(self.build_predict(Xnew), 1)
        Y_true = tf.argmax(Ynew, 1)
        correct = tf.cast(tf.equal(Y_preds, Y_true), tf.float32)
        return tf.reduce_mean(correct)
    
if __name__ == "__main__":
    batch_size = 100
    #m = Linear_model(X, Y, b=batch_size)
    m = CNN(X, Y, b=batch_size)
    m.optimize(tf.train.AdamOptimizer(), maxiter=5000)
    print m.accuracy(Xs, Ys) # should get around 88%, but gets more like 58%

    #m_100 = Linear_model(X[:batch_size, :], Y[:batch_size, :], b=100)
    m = CNN(X[:batch_size, :], Y[:batch_size, :], b=batch_size)
    m_100.optimize(tf.train.AdamOptimizer(), maxiter=5000)
    print m_100.accuracy(Xs, Ys) # similar result, but with only 100 points