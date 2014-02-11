# -*- coding: utf-8 -*-
'''
Created on 2013-12-02 14:52
@summary: For word segmenation by FNN
@author: egg
'''

import math
import numpy
import cPickle
import time

import theano
import theano.tensor as T
from pylm.MlpNgram import MlpNgram
from tagutil import formtext

class MlpWS(MlpNgram):

	def __init__(self, ndict, chunk_size = 5, n_emb=50, n_hidden=200, lr=0.05, l1_reg = 0.00, l2_reg=0.0001, batch_size=40, dropout=False, emb_file_path=None, backup_file_path=None):
		# 调用父构造方法
		super(MlpWS, self).__init__(ndict, chunk_size+1, n_emb, n_hidden, lr, l1_reg, l2_reg, batch_size, dropout, emb_file_path, backup_file_path)

		# 设置输出层大小
		self.n_out = 4
		self.chunk_size = chunk_size

	def tokens2nndata(self, train_text, train_tags):
		'''
		@summary: 转化训练数据
		'''
		train_text = "\n\n" + train_text.strip().replace("\n", "\n\n") + "\n\n"
		train_tags = train_tags.strip().replace("\n", "00")
		tidseq = [self.ndict[k] for k in train_text]
		tags = [int(tag) for tag in train_tags]

		mat_in = []
		start_index = self.chunk_size / 2
		for i in xrange(start_index, len(train_text) - start_index):
			idxs = tidseq[i - start_index : i + start_index+1]
			mat_in.append(idxs)

		
		mat_in = theano.shared(numpy.asarray(mat_in, dtype="int32"), borrow=True)
		vec_out = theano.shared(numpy.asarray(tags, dtype="int32"), borrow=True)

		return mat_in, vec_out

	def traintext(self, train_text, train_tags, test_text, test_tags, 
			epoch=100, lr_coef = -1., DEBUG=False, SAVE=False, SINDEX=1, r_init=True):

		if type(DEBUG) == bool:
			DEBUG = DEBUG and 1 or 0 
		if type(SAVE) == bool:
			SAVE = SAVE and 1 or 0 
		# token chars to token ids
		train_size = len(train_tags)
		n_batch = int(math.ceil(train_size / self.batch_size))
		
		self.setTrainData(self.tokens2nndata(train_text, train_tags))
		self.initMlp()
		test_in, test_out  = self.tokens2nndata(test_text, test_tags)
		print "MlpWS model init complete!!"

		s_time = time.clock()
		for i in xrange(epoch):

			for idx in xrange(n_batch):
				self.train_batch(idx)
				# print idx, n_batch
				# print idx * self.batch_size, train_size
		
			if DEBUG > 0:
				if (i+1) % DEBUG == 0:
					error = self.test_model(test_in.get_value(borrow=True), test_out.get_value(borrow=True))
					print "Error rate: %0.5f. Epoch: %s. Training time so far: %0.2fm" % (error, i+SINDEX, (time.clock()-s_time)/60.)
				else:
					print "Epoch %s. Training time so far is: %.2fm" % ( i+SINDEX, (time.clock()-s_time) / 60.)

			if lr_coef > 0:
				# update learning_rate
				lr = self.lr.get_value(borrow=True) * lr_coef
				self.lr.set_value(numpy.array(lr, dtype=theano.config.floatX))
				
			if SAVE > 0 and ((i+1)%SAVE == 0):
				self.savemodel("./data/MlpWS/MlpWS.model.epoch%s.chunk_size%s.n_hidden%s.dr%s.n_emb%s.in_size%s.r%s.obj" % (i+SINDEX, self.chunk_size, self.n_hidden, self.dropout, self.n_emb, self.n_in, r_init))

		e_time = time.clock()

		duration = e_time - s_time

		print "MlpNgram train over!! The total training time is %.2fm." % (duration / 60.) 	

