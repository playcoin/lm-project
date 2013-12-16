# -*- coding: utf-8 -*-
'''
Created on 2013-12-02 20:17
@summary: 带前缀的RNN
@author: egg
'''

import math
import numpy
import cPickle
import time

import theano
import theano.tensor as T
from dltools.rnnemb import RNNEMBWF2, RNNMULTIEMB
from tagutil import formtext
from RnnWS import RnnWS

class RnnWFWS(RnnWS):

	def __init__(self, ndict, n_emb, n_hidden, lr, batch_size, ext_emb=2, l2_reg=0.000001, train_emb=True, emb_file_path = None, dropout=False, truncate_step=4, backup_file_path=None):

		super(RnnWFWS, self).__init__(ndict, n_emb, n_hidden, lr, batch_size, 
			l2_reg, train_emb, emb_file_path, dropout, 
			truncate_step, backup_file_path)

		self.ext_emb = ext_emb

	def initRnn(self, no_train=False):
		'''
		@summary: Initiate RNNEMB model 
		'''
		if self.rnn is not None:
			return

		print "%s init start!" % self.__class__.__name__
		u = T.imatrix('u')
		y = T.ivector('y')
		l = T.imatrix('l')
		h_init = T.matrix('h_init')

		rng = numpy.random.RandomState(213234)
		rnn = RNNMULTIEMB(rng, 
				self.in_size,
				self.n_emb, 
				self.n_hidden, 
				self.out_size,
				self.batch_size,
				self.lr,
				self.dropout,
				params = self.rnnparams,
				embeddings = self.embvalues,
				ext_emb = self.ext_emb
			)

		self.rnn = rnn
		self.rnnparams = rnn.params

		error = rnn.errors(u,y)
		self.test_model = theano.function([u, y], error)
		print "Compile Test function complete!"
		self.rnn_prob_matrix = theano.function([u], rnn.y_prob)
		print "Compile probabilities matrix function complete!"
		self.rnn_pred = theano.function([u], rnn.y_pred)
		print "Compile predict function complete!"

	def tokens2nndata(self, train_text, train_tags=None):
		'''
		@summary: 将输入文本转化为id序列
		'''
		# 将训练文本再预处理一下，根据emb_num的个数补充后续的回车符数目
		suffix = ''.join(['\n' for x in range(self.ext_emb)])
		train_text = train_text.strip().replace("\n", "\n\n") + suffix
		tids = [self.ndict[token] for token in train_text]

		mat_in = []
		for i in xrange(0, len(tids) - self.ext_emb):
			mat_in.append(tids[i:i+self.ext_emb+1])

		mat_in = theano.shared(numpy.asarray(mat_in, dtype="int32"), borrow=True)

		if train_tags:
			suffix_tags = ''.join(['0' for x in range(self.ext_emb)])
			train_tags = train_tags.strip().replace("\n", "00")
			tags = [int(tag) for tag in train_tags]

			vec_out = theano.shared(numpy.asarray(tags, dtype="int32"), borrow=True)
	
			return mat_in.get_value(borrow=True), vec_out.get_value(borrow=True)
		else:
			return mat_in.get_value(borrow=True)

	def reshape(self, dataset, data_size):
		'''
		@summary: 将训练数据按batch_size进行reshape
		'''
		out_dataset = []
		for x in dataset:
			x = x[:data_size]
			if len(x.shape) == 1:
				s_x = x.reshape(self.batch_size, x.shape[0] / self.batch_size).T
			else:
				s_x = x.reshape(self.batch_size, x.shape[0] / self.batch_size, self.ext_emb+1)
				s_x = s_x.transpose(1, 0, 2).transpose(0, 2, 1)

			out_dataset.append(theano.shared(s_x, borrow=True))
		return out_dataset

class RnnWFWS2(RnnWS):

	def initRnn(self, no_train=False, dr_rate=0.5):
		'''
		@summary: Initiate RNNEMB model 
		'''
		if self.rnn is not None:
			return

		print "%s init start!" % self.__class__.__name__
		u = T.ivector('u')
		uf = T.ivector('uf')
		uf_2 = T.ivector('uf_2')
		y = T.ivector('y')
		l = T.imatrix('l')
		h_init = T.matrix('h_init')

		rng = numpy.random.RandomState(213234)
		rnn = RNNEMBWF2(rng, 
				self.in_size,
				self.n_emb, 
				self.n_hidden, 
				self.out_size,
				self.batch_size,
				self.lr,
				self.dropout,
				params = self.rnnparams,
				embeddings = self.embvalues, 
				dr_rate = dr_rate
			)

		self.rnn = rnn
		self.rnnparams = rnn.params

		error = rnn.errors(u,uf,uf_2,y)
		self.test_model = theano.function([u, uf, uf_2, y], error)
		print "Compile Test function complete!"
		self.rnn_prob_matrix = theano.function([u, uf, uf_2], rnn.y_prob)
		print "Compile probabilities matrix function complete!"
		self.rnn_pred = theano.function([u, uf, uf_2], rnn.y_pred)
		print "Compile predict function complete!"

	def tokens2nndata(self, train_text, train_tags=None):
		'''
		@summary: 将输入文本转化为id序列
		'''
		# 将训练文本再预处理一下, 单个回车符编程两个，回车符的标签为0
		train_text = train_text.strip() + '\n'
		train_text = train_text.replace("\n", "\n\n")
		tids = [self.ndict[token] for token in train_text]
		# input
		vec_in = theano.shared(numpy.asarray(tids[:-2], dtype="int32"), borrow=True)
		vec_in_f = theano.shared(numpy.asarray(tids[1:-1], dtype="int32"), borrow=True)
		vec_in_f_2 = theano.shared(numpy.asarray(tids[2:], dtype="int32"), borrow=True)
		# tag
		if train_tags:
			train_tags = train_tags.strip() + '\n'
			train_tags = train_tags.replace("\n", "00")
			tags = [int(tag) for tag in train_tags]
			vec_out = theano.shared(numpy.asarray(tags[:-2], dtype="int32"), borrow=True)
	
			return vec_in.get_value(borrow=True), vec_in_f.get_value(borrow=True), vec_in_f_2.get_value(borrow=True), vec_out.get_value(borrow=True)
		else:
			return vec_in.get_value(borrow=True), vec_in_f.get_value(borrow=True), vec_in_f_2.get_value(borrow=True)