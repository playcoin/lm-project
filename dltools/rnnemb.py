# -*- coding: utf-8 -*-
'''
Created on 2013-06-06 09:06
@summary: Rnn with character Embedding. Ajust embedding in the training time.
@author: egg
'''

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from rnn import RNN

class RNNEMB(RNN):
	'''
	@summary: Rnn with character Embedding. Ajust embedding in the training time.
	'''

	def __init__(self, rng, n_in, n_emb, n_h, n_out, batch_size, lr, dropout=False, params=None, activation=T.tanh, embeddings=None):
		'''
		@summary: Construct Method. Has same parameters in RNN except "embeddings"
		'''

		# RNN init 
		super(RNNEMB, self).__init__(rng, 
			n_in  = n_emb, 
			n_h   = n_h, 
			n_out = n_out, 
			batch_size = batch_size, 
			lr = lr, 
			dropout = dropout, 
			params = params, 
			activation = activation)

		# init embeddings
		if embeddings is None:
			embeddings = numpy.identity(n_in, dtype=theano.config.floatX)

		self.C = theano.shared(value=embeddings, name='C', borrow=True)

	def rnn_step(self, u_tm, h_tm):
		'''
		@summary: iter function of scan op
		
		@param u_tm: current input
		@param h_tm: last output of hidden layer
		'''
		lin_h = T.dot(self.C[u_tm], self.W_in) + T.dot(h_tm, self.W_h) + self.b_h
		h_t = self.activation(lin_h)
		return h_t

	def build_tbptt(self, seq_in, seq_l, truncate_step=5, train_emb=False):
		'''
		@summary: Build T-BPTT training theano function.
		
		@param x: theano variable for input vectors
		@param y: theano variable for correct labels
		@param h_init: theano variable for initial hidden layer value
		@param l_rate: learning_rate
		@result: 
		'''
		x = T.imatrix()
		y = T.imatrix()
		h_init = T.matrix()
		index = T.lscalar()

		self.truncate_step = truncate_step

		# output of hidden layer and output layer
		part_h, _ = theano.scan(self.rnn_step, sequences=x, outputs_info=h_init)
		if self.dropout:
			part_h = self.corrupt(part_h, 0.5)

		part_p_y_given_x, _ = theano.scan(self.rnn_softmax, sequences=part_h)
		# apply the last output of hidden layer as the next input 
		out_h = part_h[-1]
		
		#### BPTT ####
		# cost function
		l_y = y.flatten()
		part_p_y_given_x = part_p_y_given_x.reshape((l_y.shape[0], self.n_out))
		# cross-entropy loss
		part_cost = T.mean(T.nnet.categorical_crossentropy(part_p_y_given_x, l_y))
		if self.dropout:
			part_L2_sqr = (self.W_in ** 2).sum() + (self.W_h ** 2).sum() + (self.W_out ** 2).sum()
			part_cost = part_cost + 0.000001 * part_L2_sqr
		# update params
		params = [self.W_in, self.W_h, self.W_out, self.b_h, self.b_out]
		# check whether to train the embedding
		if train_emb:
			params.append(self.C)

		gparams = []
		for param in params:
			gparam = T.grad(part_cost, param)
			gparams.append(gparam)

		updates = []
		#	C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
		for param, gparam in zip(params, gparams):
			updates.append((param, param - self.lr * gparam))

		# BPTT for a truncate step. 
		self.f_part_tbptt = theano.function(inputs=[index, h_init], 
											outputs=out_h, 
											updates=updates, 
											givens = {
												x : seq_in[index : index+truncate_step],
												y : seq_l[index : index+truncate_step]
											})

	def train_tbptt(self, s_index, e_index):
		'''
		@summary: T-BPTT Algorithm
		
		@param seq_input:
		@param seq_label:
		@param learning_rate:
		@param truncate_step:
		'''
		h_init = self.h_0.get_value(borrow=True)
		# slice the sequence, do BPTT in each slice
		for j in range(s_index, e_index, self.truncate_step):
			# BPTT and reset the h_init
			h_init = self.f_part_tbptt(j, h_init)
