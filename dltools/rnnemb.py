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
		if n_emb >= n_in:	# n_emb >= n_in indicate never use embedding
			self.C = None
		elif embeddings is None:	# else if embedding is None, then init the embedding randomly
			C_values = numpy.asarray(rng.uniform(
					low=-numpy.sqrt(6. / (n_in + n_emb)),
					high=numpy.sqrt(6. / (n_in + n_emb)),
					size=(n_in, n_emb)), dtype=theano.config.floatX)

			self.C = theano.shared(value=C_values, name='C', borrow=True)
			print "Embedding random init!"
		else:
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

	def rnn_step_noemb(self, u_tm, h_tm):
		'''
		@summary: iter function of scan op
		
		@param u_tm: current input
		@param h_tm: last output of hidden layer
		'''
		lin_h = self.W_in[u_tm] + T.dot(h_tm, self.W_h) + self.b_h
		h_t = self.activation(lin_h)
		return h_t

	def errors(self, u, y):
		'''
		@summary: Errors count
		
		@param u: TensorVariable, matrix
		@param y: labels
		'''
		if not self.C:
			h, _ = theano.scan(self.rnn_step_noemb, sequences=u, outputs_info=self.h_0[0])
		else:
			h, _ = theano.scan(self.rnn_step, sequences=u, outputs_info=self.h_0[0])

		if self.dropout:
			self.y_prob = T.nnet.softmax(T.dot(h, self.W_out / 2.) + self.b_out)
		else:
			self.y_prob = T.nnet.softmax(T.dot(h, self.W_out) + self.b_out)

		self.y_pred = T.argmax(self.y_prob, axis=1)

		self.y_sort_matrix = T.sort(self.y_prob, axis=1)

		return T.mean(T.neq(self.y_pred, y))

	def build_tbptt(self, seq_in, seq_l, truncate_step=5, train_emb=False, l2_reg=0.000001):
		'''
		@summary: Build T-BPTT training theano function.
		
		@param x: theano variable for input vectors
		@param y: theano variable for correct labels
		@param h_init: theano variable for initial hidden layer value
		@result: 
		'''
		x = T.imatrix()
		y = T.imatrix()
		h_init = T.matrix()
		index = T.lscalar()

		self.truncate_step = truncate_step

		# output of hidden layer and output layer
		if not self.C:
			part_h, _ = theano.scan(self.rnn_step_noemb, sequences=x, outputs_info=h_init)
		else:
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
		# add L2 norm
		part_L2_sqr = (self.W_in ** 2).sum() + (self.W_h ** 2).sum() + (self.W_out ** 2).sum()
		part_cost = part_cost + l2_reg * part_L2_sqr
		# print l2_reg
		# if self.dropout:
		# update params
		params = [self.W_in, self.W_h, self.W_out, self.b_h, self.b_out]
		# check whether to train the embedding
		if train_emb and self.C:
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

class RNNEMBWF(RNNEMB):
	'''
	@summary: Rnn with character Embedding. Ajust embedding in the training time.
	'''

	def __init__(self, rng, n_in, n_emb, n_h, n_out, batch_size, lr, dropout=False, params=None, activation=T.tanh, embeddings=None):
		'''
		@summary: Construct Method. Has same parameters in RNN except "embeddings"
		'''

		# RNN init 
		super(RNNEMBWF, self).__init__(rng, 
			n_in, n_emb, n_h, n_out, batch_size, lr, dropout,  
			params = (params and params[:-1] or None), 
			activation = activation, embeddings = embeddings)

		# 再初始化一个前看权值矩阵
		if params is None:
			W_in_f_values = numpy.asarray(rng.uniform(
				low  = -numpy.sqrt(6.0 / (n_emb + n_h)),
				high  = numpy.sqrt(6.0 / (n_emb + n_h)),
				size = (n_emb, n_h)), dtype = theano.config.floatX
			)
			W_in_f = theano.shared(value = W_in_f_values, name='W_in', borrow=True)
		else:
			W_in_f = params[-1]

		self.W_in_f = W_in_f

	def rnn_step(self, u_tm, uf_tm, h_tm):
		'''
		@summary: iter function of scan op
		
		@param u_tm: current input
		@param h_tm: last output of hidden layer
		'''
		lin_h = T.dot(self.C[u_tm], self.W_in) + T.dot(self.C[uf_tm], self.W_in_f) + T.dot(h_tm, self.W_h) + self.b_h
		h_t = self.activation(lin_h)
		return h_t

	def errors(self, u, uf, y):
		'''
		@summary: Errors count
		
		@param u: TensorVariable, matrix
		@param y: labels
		'''
		h, _ = theano.scan(self.rnn_step, sequences=[u, uf], outputs_info=self.h_0[0])

		if self.dropout:
			self.y_prob = T.nnet.softmax(T.dot(h, self.W_out / 2.) + self.b_out)
		else:
			self.y_prob = T.nnet.softmax(T.dot(h, self.W_out) + self.b_out)

		self.y_pred = T.argmax(self.y_prob, axis=1)

		self.y_sort_matrix = T.sort(self.y_prob, axis=1)

		return T.mean(T.neq(self.y_pred, y))

	def build_tbptt(self, seq_in, seq_in_f, seq_l, truncate_step=5, train_emb=False, l2_reg=0.000001):
		'''
		@summary: Build T-BPTT training theano function.
		
		@param x: theano variable for input vectors
		@param y: theano variable for correct labels
		@param h_init: theano variable for initial hidden layer value
		@result: 
		'''
		x = T.imatrix()
		xf = T.imatrix()
		y = T.imatrix()
		h_init = T.matrix()
		index = T.lscalar()

		self.truncate_step = truncate_step

		# output of hidden layer and output layer
		part_h, _ = theano.scan(self.rnn_step, sequences=[x, xf], outputs_info=h_init)
			

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
		# add L2 norm
		part_L2_sqr = (self.W_in ** 2).sum() + (self.W_h ** 2).sum() + (self.W_out ** 2).sum()
		part_cost = part_cost + l2_reg * part_L2_sqr
		# print l2_reg
		# if self.dropout:
		# update params
		params = [self.W_in, self.W_in_f, self.W_h, self.W_out, self.b_h, self.b_out]
		# check whether to train the embedding
		if train_emb and self.C:
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
												xf : seq_in_f[index : index+truncate_step],
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