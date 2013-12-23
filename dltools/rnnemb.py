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

	def __init__(self, rng, n_in, n_emb, n_h, n_out, batch_size, lr, 
		dr_rate=0.5, emb_dr_rate=0., params=None, embeddings=None, activation=T.tanh):
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
			dropout = (dr_rate > 0.01), 
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

		self.update_params = [self.W_in, self.W_h, self.W_out, self.b_h, self.b_out]
		self.norm_params = [self.W_in, self.W_h, self.W_out]
		self.dr_rate = dr_rate
		self.emb_dr = emb_dr_rate > 0.01
		self.emb_dr_rate = emb_dr_rate
		self.inv_emb_dr = 1. / (1.-emb_dr_rate)
		self.inp_sym = T.imatrix()


	def rnn_step(self, u_tm, h_tm):
		'''
		@summary: iter function of scan op
		
		@param u_tm: current input
		@param h_tm: last output of hidden layer
		'''
		if self.emb_dr:
			lin_h = T.dot(u_tm, self.W_in / self.inv_emb_dr)
		else:
			lin_h = T.dot(u_tm, self.W_in)

		lin_h += T.dot(h_tm, self.W_h) + self.b_h
		h_t = self.activation(lin_h)
		return h_t

	def rnn_step_noemb(self, u_tm, h_tm):

		lin_h = self.W_in[u_tm] + T.dot(h_tm, self.W_h) + self.b_h
		h_t = self.activation(lin_h)
		return h_t

	def rnn_step_emb_dr(self, u_tm, h_tm):

		lin_h = T.dot(u_tm, self.W_in) + T.dot(h_tm, self.W_h) + self.b_h
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
			h, _ = theano.scan(self.rnn_step, sequences=self.C[u], outputs_info=self.h_0[0])

		if self.dropout:
			inv_dr = 1. / (1 - self.dr_rate)
			self.y_prob = T.nnet.softmax(T.dot(h, self.W_out / inv_dr) + self.b_out)
		else:
			self.y_prob = T.nnet.softmax(T.dot(h, self.W_out) + self.b_out)

		self.y_pred = T.argmax(self.y_prob, axis=1)

		self.y_sort_matrix = T.sort(self.y_prob, axis=1)

		return T.mean(T.neq(self.y_pred, y))

	def build_tbptt(self, seq_in, seq_l, truncate_step=5, train_emb=False, l2_reg=0.000001):
		'''
		@summary: Build T-BPTT training theano function. 

		@param seq_in: the input data. theano shared variables, for GPU computing. 
		@param seq_l: the labels. theano shared variables, for GPU computing
		'''
		x = self.inp_sym
		y = T.imatrix()
		h_init = T.matrix()
		index = T.lscalar()

		# output of hidden layer and output layer
		if not self.C:
			part_h, _ = theano.scan(self.rnn_step_noemb, sequences=x, outputs_info=h_init)
		elif self.emb_dr:
			nx = self.corrupt(self.C[x], self.emb_dr_rate)
			part_h, _ = theano.scan(self.rnn_step_emb_dr, sequences=nx, outputs_info=h_init)
		else:
			part_h, _ = theano.scan(self.rnn_step, sequences=self.C[x], outputs_info=h_init)
			
		

		if self.dropout:
			part_h = self.corrupt(part_h, self.dr_rate)

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
		part_L2_sqr = 0.
		for w in self.norm_params:
			part_L2_sqr += (x ** 2).sum()

		part_cost = part_cost + l2_reg * part_L2_sqr

		params = self.update_params
		# check whether to train the embedding
		if train_emb and self.C:
			params.append(self.C)
		gparams = []
		for param in params:
			gparam = T.grad(part_cost, param)
			gparams.append(gparam)
		updates = []
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

		self.truncate_step = truncate_step

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

class RNNMULTIEMB(RNNEMB):
	'''
	@summary: Rnn with character Embedding. Ajust embedding in the training time.
	'''

	def __init__(self, rng, n_in, n_emb, n_h, n_out, batch_size, lr, 
		dr_rate=0.5, emb_dr_rate=0., ext_emb=2, params=None, embeddings=None, activation=T.tanh):
		'''
		@summary: Construct Method. Has same parameters in RNN except "embeddings"
		'''

		# RNN init 
		super(RNNMULTIEMB, self).__init__(rng, n_in, n_emb, n_h, n_out, batch_size, lr,  
			dr_rate=dr_rate, emb_dr_rate=emb_dr_rate,
			params = (params and params[:-ext_emb] or None), 
			embeddings = embeddings, activation = activation)

		# 再初始化一个前看权值矩阵
		if ext_emb < 1:
			return

		ext_W_ins = []
		for i in range(ext_emb):
			w = params and params[i - ext_emb] or None # 把尾巴上的参数读进来
			if w is None:
				values = numpy.asarray(rng.uniform(
					low  = -numpy.sqrt(6.0 / (n_emb + n_h)),
					high  = numpy.sqrt(6.0 / (n_emb + n_h)),
					size = (n_emb, n_h)), dtype = theano.config.floatX
				)
				w = theano.shared(value = values, name='W_in_%d' % i, borrow=True)

			ext_W_ins.append(w)

		self.params.extend(ext_W_ins)
		self.update_params.extend(ext_W_ins)
		self.norm_params.extend(ext_W_ins)
		self.ext_W_ins = ext_W_ins
		self.ext_emb = ext_emb

		self.inp_sym = T.itensor3()

	def rnn_step(self, u_tm, h_tm):
		# 主要是为了在测试的时候也是对的...
		if self.emb_dr:
			lin_h = T.dot(u_tm[0], self.W_in / self.inv_emb_dr) + T.dot(h_tm, self.W_h) + self.b_h
			# 计算补充的embeddings
			for i in range(1, self.ext_emb+1):
				lin_h += T.dot(u_tm[i], self.ext_W_ins[i-1] / self.inv_emb_dr)
		else:
			lin_h = T.dot(u_tm[0], self.W_in) + T.dot(h_tm, self.W_h) + self.b_h
			# 计算补充的embeddings
			for i in range(1, self.ext_emb+1):
				lin_h += T.dot(u_tm[i], self.ext_W_ins[i-1])


		h_t = self.activation(lin_h)
		return h_t

	def rnn_step_emb_dr(self, u_tm, h_tm):

		lin_h = T.dot(u_tm[0], self.W_in) + T.dot(h_tm, self.W_h) + self.b_h
		# 计算补充的embeddings
		for i in range(1, self.ext_emb+1):
			lin_h += T.dot(u_tm[i], self.ext_W_ins[i-1])
		
		h_t = self.activation(lin_h)
		return h_t