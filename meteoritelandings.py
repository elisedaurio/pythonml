# Use https://data.nasa.gov/Space-Science/Meteorite-Landings/gh4g-9sfh
# To learn about landings
# 
# The goal here, is to see how well can predict how massive a meteorite will be based on it's lat and long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# The meteorite data is stored in CSV, so grab it using text line reader
inputfile = tf.train.string_input_producer(["cleanlandings.csv"])

# Call reader
reader = tf.TextLineReader()
key, value = reader.read(inputfile)

# Handle the defaults for the columns
# 
# This data has the following columns: id, nametype, mass, fall, reclat, reclong
# Features are reclat, reclong
# 
# Data 
#
# For Fall: Fall = 1
#			Found = 0
#
# For NameType: Valid = 1
# 				Relict = 0
# We are trying to determine mass
record_defaults = [[1],['Invalid'],[1.0],['Unknown'],[1.0],[1.0]]
col1, col2, col3, col4, col5, col6, = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col5, col6])

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for i in range (1000):
		# Grab one sample
		example, recid, recfall, recmass = sess.run([features, col1, col2, col3])
		print (recid, recfall, recmass, example)

	coord.request_stop()
	coord.join(threads)


# Build the variables
reclat = tf.placeholder(tf.float32)
reclong = tf.placeholder(tf.float32)

recmass = tf.placeholder(tf.float32)

#y = tf.matmul(reclat)

# use cross entropy
#cross_entropy = tf.reduce_mean(
#  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# TODO 
# Need to determine how to use a NN and shape it to the fact that we have a certain min and max lat/long
#

# Reads in the data well enough
# 


# Use this NN code to try and make it fit
#
# Start new NN code
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])

# x_image = tf.reshape(x, [-1,28,28,1])

# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)

# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])

# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# End NN Code
