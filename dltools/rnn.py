# -*- coding: utf-8 -*-
'''
Created on 2013-05-05 19:58
@summary: RNN model
@author: Playcoin
'''

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

class RNN(object):
	'''
	@summary: RNN model
	'''

	def __init__(self, rng, input, n_in, n_h, n_out, batch_size, params=None, activation=T.tanh):
		'''
		@summary: Initial params and theano variable
		
		@param rng:
		@param input:
		@param n_in:
		@param n_h:
		@param n_out:
		@param params:
		@param activation:
		@result: 
		'''

		W_in, W_h, W_out, b_h, b_out, h_0 = params or (None, None, None, None, None, None)

		# If there is a 'None' in params, initial all params anyway.
		if None in (W_in, W_h, W_out, b_h, b_out, h_0):
			# Weight initial
			W_in_values = numpy.asarray(rng.uniform(
				low  = -numpy.sqrt(6.0 / (n_in + n_out)),
				high  = numpy.sqrt(6.0 / (n_in + n_out)),
				size = (n_in, n_h)), dtype = theano.config.floatX
			)
			W_h_values = numpy.asarray(rng.uniform(
				low  = -numpy.sqrt(6.0 / (n_in + n_out)),
				high  = numpy.sqrt(6.0 / (n_in + n_out)),
				size = (n_h, n_h)), dtype = theano.config.floatX
			)
			W_out_values = numpy.asarray(rng.uniform(
				low  = -numpy.sqrt(6.0 / (n_in + n_out)),
				high  = numpy.sqrt(6.0 / (n_in + n_out)),
				size = (n_h, n_out)), dtype = theano.config.floatX
			)
			# if the activation is sigmoid, the W_in_values should multiply 4
			if activation == theano.tensor.nnet.sigmoid:
				W_in_values *= 4
				W_h_values *= 4
				W_out_values *= 4

			W_in = theano.shared(value = W_in_values, name='W_in', borrow=True)
			W_h = theano.shared(value = W_h_values, name='W_h', borrow=True)
			W_out = theano.shared(value = W_out_values, name='W_out', borrow=True)

			# biases and h_0 initialed by 0.
			b_h_values = numpy.zeros((n_h,), dtype = theano.config.floatX)
			b_out_values = numpy.zeros((n_out,), dtype = theano.config.floatX)
			h_0_values = numpy.zeros((batch_size, n_h), dtype = theano.config.floatX)

			b_h = theano.shared(value = b_h_values, name='b_h', borrow=True)
			b_out = theano.shared(value = b_out_values, name='b_out', borrow=True)
			h_0 = theano.shared(value = h_0_values, name='h_0', borrow=True)

		self.W_in = W_in
		self.W_h = W_h
		self.W_out = W_out
		self.b_h = b_h
		self.b_out = b_out
		self.h_0 = h_0
		self.input = input
		self.activation = activation
		self.batch_size = batch_size
		self.n_out = T.as_tensor_variable(n_out)

		# self.h and self.p_y_given_x are for training function
		self.h, _ = theano.scan(self.rnn_step, sequences=input, outputs_info=dict(initial=self.h_0, taps=[-1]))
		self.p_y_given_x, _ = theano.scan(self.rnn_softmax, sequences=self.h)

		self.params = [self.W_in, self.W_h, self.W_out, self.b_h, self.b_out, self.h_0]

	def rnn_step(self, u_tm, h_tm):
		'''
		@summary: iter function of scan op
		
		@param u_tm: current input
		@param h_tm: last output of hidden layer
		'''
		lin_h = T.dot(u_tm, self.W_in) + T.dot(h_tm, self.W_h) + self.b_h
		h_t = self.activation(lin_h)
		return h_t

	def rnn_softmax(self, h):
		'''
		@summary: iter function to calculate softmax of outputlayer
		
		@param h:
		@result: 
		'''
		return T.nnet.softmax(T.dot(h, self.W_out) + self.b_out)

	def loss_NLL(self, y):
		'''
		@summary: Loss function of BPTT
		
		@param y: labels
		'''
		y_tmp = self.p_y_given_x.reshape((self.input.shape[0] * self.input.shape[1], 10))
		return -T.mean(T.log(y_tmp)[T.arange(y.shape[0]), y])

	def errors(self, u, y):
		'''
		@summary: Errors count
		
		@param u: TensorVariable, matrix
		@param y: labels
		'''
		h, _ = theano.scan(self.rnn_step, sequences=u, outputs_info=self.h_0[0])
		self.y_prob = T.nnet.softmax(T.dot(h, self.W_out) + self.b_out)
		self.y_pred = T.argmax(self.y_prob, axis=1)

		self.y_sort_matrix = T.sort(self.y_prob, axis=1)

		return T.mean(T.neq(self.y_pred, y))

	def build_tbptt(self, x, y, h_init, l_rate, truncate_step=5):
		'''
		@summary: Build T-BPTT training theano function.
		
		@param x: theano variable for input matrix
		@param y: theano variable for correct labels
		@param h_init: theano variable for initial hidden layer value
		@param l_rate: learning_rate
		@result: 
		'''

		# output of hidden layer and output layer
		part_h, _ = theano.scan(self.rnn_step, sequences=x, outputs_info=h_init)
		part_p_y_given_x, _ = theano.scan(self.rnn_softmax, sequences=part_h)
		
		# apply the laster output of hidden layer as the next input 
		out_h = part_h[-1]
		
		#### BPTT ####
		# cost function
		part_p_y_given_x = part_p_y_given_x.reshape((y.shape[0], x.shape[2]))
		part_cost = -T.mean(T.log(part_p_y_given_x)[T.arange(y.shape[0]), y])
		# update params
		params = [self.W_in, self.W_h, self.W_out, self.b_h, self.b_out]

		gparams = []
		for param in params:
			gparam = T.grad(part_cost, param)
			gparams.append(gparam)

		updates = []
		#	C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
		for param, gparam in zip(params, gparams):
			updates.append((param, param - l_rate * gparam))

		# BPTT for a truncate step. 
		self.f_part_tbptt = theano.function(inputs=[x, y, h_init], outputs=out_h, updates=updates)
		self.truncate_step = truncate_step

		return self.f_part_tbptt

	def train_tbptt(self, seq_input, seq_label):
		'''
		@summary: T-BPTT Algorithm
		
		@param seq_input:
		@param seq_label:
		@param learning_rate:
		@param truncate_step:
		'''
		h_init = self.h_0.get_value(borrow=True)
		# slice the sequence, do BPTT in each slice
		for j in range(0, len(seq_input), self.truncate_step):
			# slice
			part_in = seq_input[j:j+self.truncate_step]
			part_y = seq_label[j*self.batch_size:(j+self.truncate_step)*self.batch_size]
			# BPTT and reset the h_init
			h_init = self.f_part_tbptt(part_in, part_y, h_init)
