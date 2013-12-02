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
from dltools.rnnemb import RNNEMBWF
from tagutil import formtext
from RnnWS import RnnWS

class RnnWFWS(RnnWS):

	def __init__(self, ndict, n_emb, n_hidden, lr, batch_size, l2_reg=0.000001, train_emb=True, emb_file_path = None, dropout=False, truncate_step=4, backup_file_path=None):

		super(RnnWFWS, self).__init__(ndict, n_emb, n_hidden, lr, batch_size, l2_reg, train_emb, emb_file_path, dropout, truncate_step, backup_file_path)

	def initRnn(self, no_train=False):
		'''
		@summary: Initiate RNNEMB model 
		'''
		if self.rnn is not None:
			return

		print "RnnWS init start!"
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

		probs = rnn.y_prob[T.arange(y.shape[0]), y]
		self.rnn_prob = theano.function([u, uf, y], probs)
		print "Compile likelihood function complete!"

		self.rnn_pred = theano.function([u, uf], rnn.y_pred)
		print "Compile predict function complete!"

	def tokens2nndata(self, train_text, train_tags):
		'''
		@summary: 将输入文本转化为id序列
		'''
		# 将训练文本再预处理一下, 单个回车符编程两个，回车符的标签为0
		train_text = train_text.replace("\n", "\n\n")
		train_tags = train_tags.replace("\n", "00")

		tids = [self.ndict[token] for token in train_text]
		tags = [int(tag) for tag in train_tags]


		vec_in = theano.shared(numpy.asarray(tids[:-1], dtype="int32"), borrow=True)
		vec_in_f = theano.shared(numpy.asarray(tids[1:], dtype="int32"), borrow=True)
		vec_out = theano.shared(numpy.asarray(tags[:-1], dtype="int32"), borrow=True)

		return vec_in.get_value(borrow=True), vec_in_f.get_value(borrow=True), vec_out.get_value(borrow=True)

	def reshape(self, in_data, in_data_f, l_data):
		s_in = in_data.reshape(self.batch_size, in_data.shape[0] / self.batch_size).T
		s_in_f = in_data_f.reshape(self.batch_size, in_data.shape[0] / self.batch_size).T
		s_l = l_data.reshape(self.batch_size,  l_data.shape[0] / self.batch_size).T
		return theano.shared(s_in, borrow=True), theano.shared(s_in_f, borrow=True), theano.shared(s_l, borrow=True)

	def traintext(self, train_text, train_tags, test_text, test_tags, 
			sen_slice_length=4, epoch=200, lr_coef = -1., 
			DEBUG = False, SAVE=False, SINDEX=1, r_init=False):

		self.initRnn()

		seq_size = len(train_text)

		# variables for slice sentence
		sentence_length = sen_slice_length * self.truncate_step
		half_sen_length = sentence_length / 2
		data_slice_size = sentence_length * self.batch_size
		data_size = seq_size / data_slice_size * data_slice_size

		mat_in_total, mat_in_f_total, label_total = self.tokens2nndata(train_text, train_tags)
		re_mat_in, re_mat_in_f, re_label = self.reshape(mat_in_total[:data_size], mat_in_f_total[:data_size], label_total[:data_size])
		self.rnn.build_tbptt(re_mat_in, re_mat_in_f, re_label, self.truncate_step, self.train_emb, l2_reg=self.l2_reg)
		print "Compile Truncate-BPTT Algorithm complete!"

		# for test
		t_in, t_in_f, t_l = self.tokens2nndata(test_text, test_tags)

		# total sentence length after reshape
		total_sent_len = data_size / self.batch_size
		print "Actural training data size: %s, with total sentence length: %s" % (data_size, total_sent_len)
		s_time = time.clock()
		for i in xrange(epoch):

			for j in xrange(0, total_sent_len - half_sen_length, half_sen_length):
				# reshape to matrix: [SEQ][BATCH][FRAME]
				self.rnn.train_tbptt(j, j+sentence_length)

			if DEBUG:
				err = self.test_model(t_in, t_in_f, t_l)
				e_time = time.clock()
				print "Error rate in epoch %s, is %.5f. Training time so far is: %.2fm" % ( i+SINDEX, err, (e_time-s_time) / 60.)
				# print formtext(test_text, self.rnn_pred(t_in))

			if SAVE:
				class_name = self.__class__.__name__
				self.savemodel("./data/%s/%s.model.epoch%s.n_hidden%s.ssl%s.truncstep%s.dr%s.embsize%s.in_size%s.r%s.obj" % (class_name, class_name, i+SINDEX, self.n_hidden, sen_slice_length, self.truncate_step, self.dropout, self.n_emb, self.in_size, r_init))

			if lr_coef > 0:
				# update learning_rate
				lr = self.rnn.lr.get_value() * lr_coef
				self.rnn.lr.set_value(numpy.array(lr, dtype=theano.config.floatX))

		e_time = time.clock()
		print "RnnLM train over!! The total training time is %.2fm." % ((e_time - s_time) / 60.) 