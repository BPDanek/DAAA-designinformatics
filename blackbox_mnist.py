# mnist classifier that is unknown to network
# Author: Benjamin Danek
# Heavily based off of Aymeric Damien's example: https://github.com/aymericdamien/TensorFlow-Examples/ 
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py


from __future__ import division, print_function, absolute_import

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf
from tf.layers import conv2d, max_pooling2d, dense, dropout
from tf.contrib.layers import flatten
from tf.nn import relu, softmax, sparse_softmax_cross_entropy_with_logits
from tf.train import AdamOptimizer
from tf import argmax, estimator

# training params
learning_rate = 0.001
num_steps = 2000
batch_size = 16

# network params
num_input = 784 # mnist is 28*28 image, this is the number of inputted pixels (1d)
num_classes = 10 # number of classes, 0-9
dropout = 0.25 # probability to drop a unit (neuron?)

def convnet_architecture(image_input, n_classes, dropout_rate, reuse, is_training):
	
	with tf.variable_scope('convnet_architecture', reuse=reuse):
	
		image_input = image_input['images'] # assign a name to the parameter to this def
		
		# mnist data input is a 1-D vector of 784 features, 28x28 pixels
		# must be reshaped to match picture format (height, width, channels) --> (28, 28, 1)
		shaped_image_input = tf.reshape(image_input, shape=[-1, 28, 28, 1])
		# reshape(tensor(src), shape, name(optional)); not sure why -1 is here
		
		# layers 

		# layer 1: Conv -> Pooling
		# conv2d, 32 filters, w/ kernel size 5
		conv1 = conv2d(shaped_image_input, 32, 5, activation=relu)
		# Max pooling w/ stride 2, kernel size 2
		conv1 = max_pooling2d(conv1, 2, 2)

		# layer 2: Conv -> Pooling
		# conv2d, 64 filters, kernel size 3
		conv2 = conv2d(conv1, 64, 3, activation=relu)
		# Max pooling w/ stride 2, kernel size 2
		conv2 = max_pooling2d(conv2d, 2, 2)

		# Flatten data into 1-D vector (this is required for input into dense)
		flat = flatten(conv2) 
		fc1 = dense(flat, 1024)
		
		# Apply dropout to large dense layer
		# dropout is toggled by "is_training" boolean
		fc1_dropout = dropout(fc1, rate=dropout, training=is_training)

		output = dense(fc1_dropout, n_classes)
	
	return output

# Define model function using the tf estimator template:
# https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
def model_fn(features, labels, mode):
	
	# Build the neural network
	    # dropout activates only during training, so we need to create 2 separate
	    # computation graphs that share the same weights
	logits_train = convnet_architecture(images, 
					    num_classes, 
					    dropout_rate, 
					    reuse=False, 
					    is_training=True)
	logits_test = convnet_architecture(images,
					   num_classes,
					   dropout_rate,
					   reuse=True, 
					   is_training=False)
	
	# Predictions
	pred_classes = argmax(logits_test, axis=1)
	pred_probas = softmax(logits_test)

	# If in prediction mode, early return (before training occurs
	# Predict denotes inference mode 
	if mode == tf.estimator.ModeKeys.Predict: 
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
	
	# Define loss
	# - reduce mean of sparse softmax xent w/ logits
	# 	- logits comes from logits_train
	#	- labels are cast from whatever form they come in to int32
	loss_operation = tf.reduce_mean(sparse_softmax_cross_entropy_with_logits(
		logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))

	# Define optimizer, with global lr
	optimizer = AdamOptimizer(learning_rate=learning_rate)
	
	# training operation defined: 
	# instruction for optimizer to minimize the loss, at global step
	train_operation = optimizer.minimize(loss_operation, global_step=tf.train.get_global_step())
	
	# evaluate accuracy of the model
	accuracy_operation = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

	# TF estimators require estimator specification that elaborates different operations
	# for training vs. evaluating
	estimator_specs = estimator.EstimatorSpec(mode=mode,
						  predictions=prediction_classes,
						  loss=loss_operation,
						  train_op=train_operation,
						  eval_metric_ops={'accuracy': acc_op})
	
	return estimator_specs

# build the estimator 
model = estimator.Estimator(model_fn)
	
# Define the input function for training 
input_fn = estimator.inputs.numpy_input_fn(
	x={'images':mnist.test.images}, y=mnist.test.labels,
	batch_size=batch_size,
	shuffle=False)

# use estimator to 'evaluate' the method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
