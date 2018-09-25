#encoding:utf-8
from __future__ import division
import os
from os.path import join as pjoin
from inference.config import *

import sys

import tensorflow as tf


NUM_CLASSES = INPUT_SIZE*INPUT_SIZE

flags = tf.app.flags
FLAGS = flags.FLAGS

# Autoencoder Architecture Specific Flags
flags.DEFINE_integer("num_hidden_layers", 3, "Number of hidden layers")

flags.DEFINE_integer('hidden1_units', 800,
                     'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2_units', 800,
                     'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3_units', 800,
                     'Number of units in hidden layer 3.')

flags.DEFINE_integer('num_classes', INPUT_SIZE*INPUT_SIZE, 'Number of classes')

flags.DEFINE_float('pre_layer1_learning_rate', 0.001,
                   'Initial learning rate.')
flags.DEFINE_float('pre_layer2_learning_rate', 0.0003,
                   'Initial learning rate.')
flags.DEFINE_float('pre_layer3_learning_rate', 0.0001,
                   'Initial learning rate.')

flags.DEFINE_float('noise_1', 0.50, 'Rate at which to set pixels to 0')
flags.DEFINE_float('noise_2', 0.50, 'Rate at which to set pixels to 0')
flags.DEFINE_float('noise_3', 0.50, 'Rate at which to set pixels to 0')

# Constants
flags.DEFINE_integer('seed', 1234, 'Random seed')

flags.DEFINE_integer('batch_size', BATCH_SIZE,
                     'Batch size. Must divide evenly into the dataset sizes.')

flags.DEFINE_float('supervised_learning_rate', 0.1,
                   'Supervised initial learning rate.')

flags.DEFINE_integer('pretraining_epochs', 60,
                     "Number of training epochs for pretraining layers")
flags.DEFINE_integer('finetuning_epochs', 56,
                     "Number of training epochs for "
                     "fine tuning supervised step")

flags.DEFINE_float('zero_bound', 1.0e-9,
                   'Value to use as buffer to avoid '
                   'numerical issues at 0')
flags.DEFINE_float('one_bound', 1.0 - 1.0e-9,
                   'Value to use as buffer to avoid numerical issues at 1')

flags.DEFINE_float('flush_secs', 120, 'Number of seconds to flush summaries')

# TensorBoard
flags.DEFINE_boolean('no_browser', True,
                     'Whether to start browser for TensorBoard')

# Python
flags.DEFINE_string('python', sys.executable,
                    'Path to python executable')
