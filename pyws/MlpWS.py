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

	def tokens2nndata(self, train_text, train_tags):
		'''
		@summary: 转化训练数据
		'''
		# 分句
		lines = train_text.split('\n')
		taglines = train_tags.split('\n')
		mat_in = []
		vec_out = []
		# 逐句处理
		for i in range(len(lines)):
			line = "\n\n%s\n\n" % lines[i]
			lids = [self.ndict[x] for x in line]
			# 变成向量
			for j in range(2, len(lids) - 2):
				idxs = lids[j-2:j+3]
				mat_in.append(idxs)

			vec_out.extend(taglines[i])

		mat_in = theano.shared(numpy.asarray(mat_in, dtype="int32"), borrow=True)
		vec_out = theano.shared(numpy.asarray(vec_out, dtype="int32"), borrow=True)

		return mat_in, vec_out

	def traintext(self, train_text, train_tags, test_text, test_tags, 
			epoch=100, lr_coef = -1., DEBUG=False, SAVE=False, SINDEX=1, r_init=True):
		# token chars to token ids
		train_size = len(train_tags) - 1
		n_batch = int(math.ceil(train_size / self.batch_size))
		
		self.setTrainData(self.tokens2nndata(train_text, train_tags))
		self.initMlp()
		test_data = self.tokens2nndata(test_text, test_tags)
		test_in = test_data[0].get_value(borrow=True)
		test_out = test_data[1].get_value(borrow=True)
		print "MlpNgram model init complete!!"

		s_time = time.clock()
		for i in xrange(epoch):

			for idx in xrange(n_batch):
				self.train_batch(idx)

			if DEBUG:
				error = self.test_model(test_in, test_out)
				print "Error rate: %0.5f. Epoch: %s. Training time so far: %0.1fm" % (error, i+SINDEX, (time.clock()-s_time)/60.)

			if SAVE:
				self.savemodel("./data/MlpWS/MlpWS.model.epoch%s.n_hidden%s.dr%s.n_emb%s.in_size%s.r%s.obj" % (i+SINDEX, self.n_hidden, self.dropout, self.n_emb, self.n_in, r_init))
			
			if lr_coef > 0:
				# update learning_rate
				lr = self.lr.get_value(borrow=True) * lr_coef
				self.lr.set_value(numpy.array(lr, dtype=theano.config.floatX))

		e_time = time.clock()

		duration = e_time - s_time

		print "MlpNgram train over!! The total training time is %.2fm." % (duration / 60.) 	

