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
from dltools.rnnemb import RNNEMBWF, RNNEMBWF2
from tagutil import formtext
from RnnWS import RnnWS

class RnnWFWS(RnnWS):

	def initRnn(self, no_train=False):
		'''
		@summary: Initiate RNNEMB model 
		'''
		if self.rnn is not None:
			return

		print "%s init start!" % self.__class__.__name__
		u = T.ivector('u')
		uf = T.ivector('uf')
		y = T.ivector('y')
		l = T.imatrix('l')
		h_init = T.matrix('h_init')

		rng = numpy.random.RandomState(213234)
		rnn = RNNEMBWF(rng, 
				self.in_size,
				self.n_emb, 
				self.n_hidden, 
				self.out_size,
				self.batch_size,
				self.lr,
				self.dropout,
				params = self.rnnparams,
				embeddings = self.embvalues
			)

		self.rnn = rnn
		self.rnnparams = rnn.params

		error = rnn.errors(u,uf,y)
		self.test_model = theano.function([u, uf, y], error)
		print "Compile Test function complete!"
		self.rnn_prob_matrix = theano.function([u, uf], rnn.y_prob)
		print "Compile probabilities matrix function complete!"
		self.rnn_pred = theano.function([u, uf], rnn.y_pred)
		print "Compile predict function complete!"

	def tokens2nndata(self, train_text, train_tags):
		'''
		@summary: 将输入文本转化为id序列
		'''
		# 将训练文本再预处理一下, 单个回车符编程两个，回车符的标签为0
		train_text = train_text.strip() + '\n'
		train_text = train_text.replace("\n", "\n\n")
		tids = [self.ndict[token] for token in train_text]
		vec_in = theano.shared(numpy.asarray(tids[:-1], dtype="int32"), borrow=True)
		vec_in_f = theano.shared(numpy.asarray(tids[1:], dtype="int32"), borrow=True)

		if train_tags:
			train_tags = train_tags.strip() + '\n'
			train_tags = train_tags.replace("\n", "00")
			tags = [int(tag) for tag in train_tags]
			vec_out = theano.shared(numpy.asarray(tags[:-1], dtype="int32"), borrow=True)
	
			return vec_in.get_value(borrow=True), vec_in_f.get_value(borrow=True), vec_out.get_value(borrow=True)
		else:
			return vec_in.get_value(borrow=True), vec_in_f.get_value(borrow=True)

class RnnWFWS2(RnnWS):

	def initRnn(self, no_train=False):
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
				embeddings = self.embvalues
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

class RnnWFWBWS(RnnWFWS2):	

	def tokens2nndata(self, train_text, train_tags=None):
		'''
		@summary: 将输入文本转化为id序列
		'''
		# 将训练文本再预处理一下, 单个回车符编程两个，回车符的标签为0
		train_text = train_text.strip()
		train_text = '\n' + train_text.replace("\n", "\n\n") + '\n'
		tids = [self.ndict[token] for token in train_text]
		# input
		vec_in = theano.shared(numpy.asarray(tids[:-2], dtype="int32"), borrow=True)
		vec_in_f = theano.shared(numpy.asarray(tids[1:-1], dtype="int32"), borrow=True)
		vec_in_f_2 = theano.shared(numpy.asarray(tids[2:], dtype="int32"), borrow=True)
		# tag
		if train_tags:
			train_tags = train_tags.strip()
			train_tags = train_tags.replace("\n", "00")
			tags = [int(tag) for tag in train_tags]
			vec_out = theano.shared(numpy.asarray(tags, dtype="int32"), borrow=True)
	
			return vec_in.get_value(borrow=True), vec_in_f.get_value(borrow=True), vec_in_f_2.get_value(borrow=True), vec_out.get_value(borrow=True)
		else:
			return vec_in.get_value(borrow=True), vec_in_f.get_value(borrow=True), vec_in_f_2.get_value(borrow=True)