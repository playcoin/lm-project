"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

	f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

	- textbooks: "Pattern Recognition and Machine Learning" -
				 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression


class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
				 activation=T.tanh, one_hot=False):
		"""
		Typical hidden layer of a MLP: units are fully-connected and have
		sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
		and the bias vector b is of shape (n_out,).

		NOTE : The nonlinearity used here is tanh

		Hidden unit activation is given by: tanh(dot(input,W) + b)

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.matrix or theano.tensor.vector
		:param input: a symbolic tensor of shape (n_examples, n_in) or (n_examples,)

		:type n_in: int
		:param n_in: dimensionality of input

		:type n_out: int
		:param n_out: number of hidden units

		:type activation: theano.Op or function
		:param activation: Non linearity to be applied in the hidden
						   layer
		"""
		self.input = input

		# `W` is initialized with `W_values` which is uniformely sampled
		# from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
		# for tanh activation function
		# the output of uniform if converted using asarray to dtype
		# theano.config.floatX so that the code is runable on GPU
		# Note : optimal initialization of weights is dependent on the
		#		activation function used (among other things).
		#		For example, results presented in [Xavier10] suggest that you
		#		should use 4 times larger initial weights for sigmoid
		#		compared to tanh
		#		We have no info for other function, so we use the same as
		#		tanh.
		if W is None:
			W_values = numpy.asarray(rng.uniform(
					low=-numpy.sqrt(6. / (n_in + n_out)),
					high=numpy.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)), dtype=theano.config.floatX)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		if not one_hot: 
			lin_output = T.dot(input, self.W) + self.b
		else:
			lin_output = self.W[input] + self.b
		self.output = (lin_output if activation is None
					   else activation(lin_output))
		# parameters of the model
		self.params = [self.W, self.b]


class MLP(object):
	"""Multi-Layer Perceptron Class

	A multilayer perceptron is a feedforward artificial neural network model
	that has one layer or more of hidden units and nonlinear activations.
	Intermediate layers usually have as activation function thanh or the
	sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
	top layer is a softamx layer (defined here by a ``LogisticRegression``
	class).
	"""

	def __init__(self, rng, input, n_in, n_hidden, n_out, hW=None, hb=None, oW=None, ob=None):
		"""Initialize the parameters for the multilayer perceptron

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
		architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
		which the datapoints lie

		:type n_hidden: int
		:param n_hidden: number of hidden units

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
		which the labels lie

		"""

		# Since we are dealing with a one hidden layer MLP, this will
		# translate into a TanhLayer connected to the LogisticRegression
		# layer; this can be replaced by a SigmoidalLayer, or a layer
		# implementing any other nonlinearity
		self.hiddenLayer = HiddenLayer(rng=rng, input=input,
									   n_in=n_in, n_out=n_hidden, W=hW, b=hb,
									   activation=T.tanh, one_hot=True)

		# The logistic regression layer gets as input the hidden units
		# of the hidden layer
		self.logRegressionLayer = LogisticRegression(
			rng=rng,
			input=self.hiddenLayer.output,
			n_in=n_hidden,
			n_out=n_out,
			W=oW,
			b=ob)

		# L1 norm ; one regularization option is to enforce L1 norm to
		# be small
		self.L1 = abs(self.hiddenLayer.W).sum() \
				+ abs(self.logRegressionLayer.W).sum()

		# square of L2 norm ; one regularization option is to enforce
		# square of L2 norm to be small
		self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
					+ (self.logRegressionLayer.W ** 2).sum()

		# negative log likelihood of the MLP is given by the negative
		# log likelihood of the output of the model, computed in the
		# logistic regression layer
		self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
		# same holds for the function computing the number of errors
		self.errors = self.logRegressionLayer.errors

		# the parameters of the model are the parameters of the two layer it is
		# made out of
		self.params = self.hiddenLayer.params + self.logRegressionLayer.params

class MLPEMB(object):
	"""Multi-Layer Perceptron Class with Embedding
	"""

	def __init__(self, rng, input, n_in, n_token, n_emb, n_hidden, n_out, dropout=False, hW=None, hb=None, oW=None, ob=None, C=None):
		"""Initialize the parameters for the multilayer perceptron

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
		architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
		which the datapoints lie

		:type n_hidden: int
		:param n_hidden: number of hidden units

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
		which the labels lie

		"""

		# look up table C
		if not C:
			C_values = numpy.asarray(rng.uniform(
					low=-numpy.sqrt(6. / (n_in + n_emb)),
					high=numpy.sqrt(6. / (n_in + n_emb)),
					size=(n_in, n_emb)), dtype=theano.config.floatX)

			C = theano.shared(value=C_values, name='C', borrow=True)

		self.C = C

		# look up embedding in C
		sub_input = C[input]

		m_input = sub_input.flatten(ndim=2)
		# Since we are dealing with a one hidden layer MLP, this will
		# translate into a TanhLayer connected to the LogisticRegression
		# layer; this can be replaced by a SigmoidalLayer, or a layer
		# implementing any other nonlinearity
		self.hiddenLayer = HiddenLayer(rng=rng, input=m_input,
									   n_in=n_token * n_emb, n_out=n_hidden, W=hW, b=hb,
									   activation=T.tanh)

		# The logistic regression layer gets as input the hidden units
		# of the hidden layer
		h_out = self.hiddenLayer.output

		self.logRegressionLayer = LogisticRegression(
			rng=rng,
			input=h_out,
			n_in=n_hidden,
			n_out=n_out,
			dropout=dropout,
			W=oW,
			b=ob)

		# L1 norm ; one regularization option is to enforce L1 norm to
		# be small
		self.L1 = abs(self.hiddenLayer.W).sum() \
				+ abs(self.logRegressionLayer.W).sum()

		# square of L2 norm ; one regularization option is to enforce
		# square of L2 norm to be small
		self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
					+ (self.logRegressionLayer.W ** 2).sum()

		# negative log likelihood of the MLP is given by the negative
		# log likelihood of the output of the model, computed in the
		# logistic regression layer
		self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
		# same holds for the function computing the number of errors
		self.errors = self.logRegressionLayer.errors

		# the parameters of the model are the parameters of the two layer it is
		# made out of
		self.params = self.hiddenLayer.params + self.logRegressionLayer.params + [self.C]