# Recreating the previous example but with a neural network
# Using the https://www.tensorflow.org/get_started/mnist/pros tutorial for backup.

# Using the same header as the original tutorial
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Load the data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Start the interactive session
sess = tf.InteractiveSession()

# Build the computational graph
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define the weights and biases
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Run a linear regression
y = tf.matmul(x,W) + b

# Initialize all variables
sess.run(tf.global_variables_initializer())

# Use cross entropy loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Define the training step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Perform the training
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Determine accuracy by first defining our prediction
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Find the actual accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Print results
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))