from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange

import logging
import time
import argparse
import os

import keras
from keras import backend
from keras.datasets import cifar10
from keras.utils import np_utils

import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils import to_categorical
from cleverhans.utils import set_log_level, setup_logger
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils_keras import cnn_cifar10_model

from cleverhans_tutorials.tutorial_models import make_basic_cnn, MLP
from cleverhans_tutorials.tutorial_models import Flatten, Linear, ReLU, Softmax

def data_cifar10():
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def define():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nb_classes',
        type=int,
        default=10,
        help='Number of classes in problem.'
    )
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
        default="./logs/cifar-10-adv%s/"%(timestr),
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

    parser.add_argument(
        '--training_eps',
        type=float,
        default=0.3,
        help='Training FGSM epsilon.'
    )

    #Flags related to adversarial example
    parser.add_argument(
        '--crafting_eps',
        type=float,
        default=0.3,
        help='Crafting FGSM epsilon.'
    )
    #Flags related to substitute
    parser.add_argument(
        '--holdout',
        type=int,
        default=150,
        help='Test set holdout for adversary.'
    )
    parser.add_argument(
        '--data_aug',
        type=int,
        default=6,
        help='Nb of substitute data augmentations.'
    )
    parser.add_argument(
        '--nb_epochs_s',
        type=int,
        default=10,
        help='Training epochs for substitute.'
    )
    parser.add_argument(
        '--lmbda',
        type=float,
        default=0.1,
        help='Lambda from arxiv.org/abs/1602.02697.'
    )
    global FLAGS
    FLAGS = parser.parse_args()

def setup_tutorial():
    """
    Helper function to check correct configuration of tf for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    return True

def prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
              nb_epochs, batch_size, learning_rate,
              rng):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for cifar
    :param y: the ouput placeholder for cifar
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param rng: numpy.random.RandomState
    :return:
    """

    # Define TF model graph (for the black-box model)
    model = cnn_cifar10_model(img_rows=32, img_cols=32, channels=3)
    predictions = model(x)
    fgsm_params = {'eps': FLAGS.training_eps, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model, sess=sess)
    predictions_adv = model(fgsm.generate(x, **fgsm_params))
    logger.info("Defined TensorFlow model graph.")

    # Train an cifar model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    model_train(sess, x, y, predictions, X_train, Y_train, verbose=False,
                args=train_params, rng=rng, predictions_adv=predictions_adv)

    # logger.info out the accuracy on legitimate data
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    logger.info('Test accuracy of adversarially trained black-box on legitimate test '
          'examples: ' + str(accuracy))

    return model, predictions, accuracy


def substitute_model(img_rows=32, img_cols=32, nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """
    input_shape = (None, img_rows, img_cols, 3)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    return MLP(layers, input_shape)


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              rng):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_sub = substitute_model()
    preds_sub = model_sub(x)
    logger.info("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        logger.info("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                    init_all=False, verbose=False, args=train_params,
                    rng=rng)

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            logger.info("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads, lmbda)

            logger.info("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            eval_params = {'batch_size': batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def cifar_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=10, batch_size=128,
                   learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
                   nb_epochs_s=10, lmbda=0.1):
    """
    cifar tutorial for the black-box attack from arxiv.org/abs/1602.02697
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Perform tutorial setup
    assert setup_tutorial()


    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        logger.info("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session
    sess = tf.Session()
    keras.backend.set_session(sess)
    # Get cifar data
    X_train, Y_train, X_test, Y_test = data_cifar10()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[holdout:]
    Y_test = Y_test[holdout:]

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Seed random number generator so tutorial is reproducible
    rng = np.random.RandomState([2017, 8, 30])

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    logger.info("Preparing the black-box model.")
    prep_bbox_out = prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
                              nb_epochs, batch_size, learning_rate,
                              rng=rng)
    model, bbox_preds, accuracies['bbox'] = prep_bbox_out

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    logger.info("Training the substitute model.")
    train_sub_out = train_sub(sess, x, y, bbox_preds, X_sub, Y_sub,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda, rng=rng)
    model_sub, preds_sub = train_sub_out

    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_sub, X_test, Y_test, args=eval_params)
    accuracies['sub'] = acc

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': FLAGS.eps, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute
    eval_params = {'batch_size': batch_size}
    x_adv_sub = fgsm.generate(x, **fgsm_par)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, model(x_adv_sub), X_test, Y_test,
                          args=eval_params)
    logger.info('Test accuracy of oracle on adversarial examples generated '
          'using the substitute: ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex'] = accuracy
    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, model_sub(x_adv_sub), X_test, Y_test,
                          args=eval_params)
    logger.info('Test accuracy of substitute on adversarial examples generated '
          'using the substitute: ' + str(accuracy))
    accuracies['sub_on_sub_adv_ex'] = accuracy

    return accuracies


def main(argv=None):
    define()
    if (FLAGS.summaries_dir and not os.path.exists(FLAGS.summaries_dir)):
        os.makedirs(FLAGS.summaries_dir)
    setup_logger('log', stream=True, log_file=FLAGS.summaries_dir+"results.txt")
    global logger
    logger = logging.getLogger('log')
    logger.info(FLAGS)
    cifar_blackbox(nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                   data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                   lmbda=FLAGS.lmbda)


if __name__ == '__main__':
    tf.app.run()
