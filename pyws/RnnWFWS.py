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
	'''
	@summary: 带后缀的分词器，后缀数量有ext_num变量控制
	'''


	def __init__(self, ndict, n_emb, n_hidden, lr, batch_size, 
		ext_emb=2, l2_reg=0.000001, train_emb=True, emb_file_path = None, 
		dropout=False, truncate_step=4, backup_file_path=None):

		super(RnnWFWS, self).__init__(ndict, n_emb, n_hidden, lr, batch_size, 
			l2_reg, train_emb, emb_file_path, dropout, 
			truncate_step, backup_file_path)

		self.ext_emb = ext_emb
		# 通过前后缀的文本来调整输入的偏移量
		self.train_preffix = ''
		self.train_suffix = ''.join(['\n' for x in range(ext_emb)])

	def initRnn(self, no_train=False, dr_rate=0.5):
		'''
		@summary: Initiate RNNEMB model 
		'''
		if self.rnn is not None:
			return

		print "%s with %d extra embs init start! " % (self.__class__.__name__, self.ext_emb)
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
				dr_rate = dr_rate,
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
		train_text = self.train_preffix + train_text.strip().replace("\n", "\n\n") + self.train_suffix
		tids = [self.ndict[token] for token in train_text]

		mat_in = []
		for i in xrange(0, len(tids) - self.ext_emb):
			mat_in.append(tids[i:i+self.ext_emb+1])

		mat_in = theano.shared(numpy.asarray(mat_in, dtype="int32"), borrow=True)

		if train_tags:
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

class RnnWFWS2(RnnWFWS):
	'''
	@summary: 带两个后缀的分词器
	'''

	def __init__(self, *args, **kwargs):
		super(RnnWFWS2, self).__init__(*args, **kwargs):

		self.ext_emb = 2
		self.train_preffix = ''
		self.train_suffix = '\n\n'


class RnnWBWF2WS(RnnWFWS):
	'''
	@summary: 带一个后缀和两个前缀的分词器
	'''

	def __init__(self, *args, **kwargs):
		super(RnnWBWF2WS, self).__init__(*args, **kwargs):

		self.ext_emb = 3
		self.train_preffix = '\n'
		self.train_suffix = '\n\n'

