"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
import time
import argparse
from cleverhans.utils_mnist import data_cifar10
from cleverhans.utils_tf import model_gpdnn_train, model_eval, model_wrap_gp, model_gpdnn_eval
from cleverhans.attacks import FastGradientMethodGP
from cleverhans_tutorials.tutorial_models import make_gp_cifar10_cnn
from cleverhans.utils import AccuracyReport, set_log_level, setup_logger
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data

import os

FLAGS = flags.FLAGS

"""
CleverHans is intended to supply attacks and defense, not models.
Users may apply CleverHans to many different kinds of models.
In this tutorial, we show you an example of the kind of model
you might build.
"""

def define():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Size of training batches.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for training.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default="./logs/%s/"%(timestr),
        help='Summaries/Tensorboard Location'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='checpoint file'
    )

    #Flags related to oracle
    parser.add_argument(
        '--nb_epochs',
        type=int,
        default=10,
        help='Number of epochs to train model.'
    )

    #Flags related to adversarial example
    parser.add_argument(
        '--eps',
        type=float,
        default=0.3,
        help='FGSM epsilon.'
    )
    parser.add_argument(
        '--clean_train',
        dest='clean_train',
        action='store_true'
    )
    parser.add_argument(
        '--no_clean_train',
        dest='clean_train',
        action='store_false'
    )
    parser.set_defaults(clean_train=True)
    parser.add_argument(
        '--backprop_through_attack',
        dest='backprop_through_attack',
        action='store_true'
    )
    parser.add_argument(
        '--no_backprop_through_attack',
        dest='backprop_through_attack',
        action='store_false'
    )
    parser.set_defaults(backprop_through_attack=False)
    global FLAGS
    FLAGS = parser.parse_args()

def get_mnist():
    mnist = mnist_input_data.read_data_sets(os.path.join(os.path.dirname(__file__), "mnist_data/"), one_hot=False, validation_size=0)

    norm_data = lambda img_in: 2.*img_in - 1.
    add_extra_dim = lambda x: x[:, np.newaxis]

    return (norm_data(mnist.train.images), add_extra_dim(mnist.train.labels),
            norm_data(mnist.validation.images), add_extra_dim(mnist.validation.labels),
            norm_data(mnist.test.images), add_extra_dim(mnist.test.labels))


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param testing: if true, complete an AccuracyReport for unit tests
      to verify that performance is adequate
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    #tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    tf_graph = tf.Graph()
    # Create TF session
    sess = tf.Session(graph=tf_graph)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_cifar10()
    Y_train = np.argmax(Y_train, axis=1)
    Y_test_OneHot = Y_test
    Y_test = np.argmax(Y_test, axis=1)
    # Define input TF placeholder
    with tf_graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 32*32*3])
        x_reshaped = tf.reshape(x,[-1, 32, 32, 3])
        y = tf.placeholder(tf.int32, shape=[None])

    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    fgsm_params = {'eps': FLAGS.eps, 'y': Y_test_OneHot}
    rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        with tf_graph.as_default():
            model = make_gp_cifar10_cnn(num_h=100, input_shape=[None, 32, 32, 3])
            h = model.get_logits(x_reshaped)
            nn_vars = tf.global_variables()  # only nn variables exist up to now.
        sess.run(tf.variables_initializer(nn_vars))

        #Wrap GP layer around
        gp_model, train_step, preds = model_wrap_gp(sess, tf_graph, x, y, X_train, Y_train, h, args=train_params, shape=32, channel=3)


        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            with tf_graph.as_default():
                eval_params = {'batch_size': batch_size}
                acc, ll = model_gpdnn_eval(
                    sess, gp_model, x_reshaped, y, h, preds, X_test, Y_test, args=eval_params)
                report.clean_train_clean_eval = acc
                assert X_test.shape[0] == test_end - test_start, X_test.shape
                logger.info('Test accuracy on legitimate examples: %0.4f' % acc)
        model_gpdnn_train(sess, x, y, h, X_train, Y_train, train_step, evaluate=evaluate,
                    args=train_params, rng=rng, shape=32, channel=3)

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        with tf_graph.as_default():
            fgsm = FastGradientMethodGP(model, preds, sess=sess)

            adv_x = fgsm.generate(x_reshaped, **fgsm_params)
            if not backprop_through_attack:
                # For the fgsm attack used in this tutorial, the attack has zero
                # gradient so enabling this flag does not change the gradient.
                # For some other attacks, enabling this flag increases the cost of
                # training, but gives the defender the ability to anticipate how
                # the atacker will change their strategy in response to updates to
                # the defender's parameters.
                adv_x = tf.stop_gradient(adv_x)
            preds_adv = preds

        # Evaluate the accuracy of the MNIST model on adversarial examples
            eval_par = {'batch_size': batch_size}
            feed_dict = {x_reshaped: X_test}
            X_test_adv = sess.run(adv_x, feed_dict=feed_dict)
            acc, ll = model_gpdnn_eval(sess, gp_model, x_reshaped, y, h, preds_adv, X_test_adv, Y_test, args=eval_par)
            logger.info('Test accuracy on adversarial examples: %0.4f\n' % acc)
            report.clean_train_adv_eval = acc

    return report


def main(argv=None):
    define()
    if (FLAGS.summaries_dir and not os.path.exists(FLAGS.summaries_dir)):
        os.makedirs(FLAGS.summaries_dir)
    setup_logger('log', stream=True, log_file=FLAGS.summaries_dir+"cifar10_gpdnn_results.txt")
    global logger
    logger = logging.getLogger('log')
    logger.info(FLAGS)
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack)


if __name__ == '__main__':
    tf.app.run()
