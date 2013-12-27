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
from dltools.rnnemb import RNNMULTIEMB
from tagutil import formtext
from RnnWS import RnnWS

class RnnWFWS(RnnWS):
	'''
	@summary: 带后缀的分词器，后缀数量有ext_num变量控制
	'''

	def __init__(self, ndict, n_emb, n_hidden, lr, batch_size, 
		ext_emb=2, l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5, emb_dr_rate = 0.,
		emb_file_path = None, backup_file_path=None):

		super(RnnWFWS, self).__init__(ndict, n_emb, n_hidden, lr, batch_size, 
			l2_reg, truncate_step, train_emb, dr_rate, emb_dr_rate, 
			emb_file_path, backup_file_path)

		self.ext_emb = ext_emb
		# 通过前后缀的文本来调整输入的偏移量
		self.train_preffix = ''
		self.train_suffix = ''.join(['\n' for x in range(ext_emb)])

	def initRnn(self, no_train=False):
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
				dr_rate = self.dr_rate,
				emb_dr_rate = self.emb_dr_rate,
				ext_emb = self.ext_emb,
				params = self.rnnparams,
				embeddings = self.embvalues,
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
		super(RnnWFWS2, self).__init__(*args, **kwargs)

		self.ext_emb = 2
		self.train_preffix = ''
		self.train_suffix = '\n\n'


class RnnWBWF2WS(RnnWFWS):
	'''
	@summary: 带一个后缀和两个前缀的分词器
	'''

	def __init__(self, *args, **kwargs):
		super(RnnWBWF2WS, self).__init__(*args, **kwargs)

		self.ext_emb = 3
		self.train_preffix = '\n'
		self.train_suffix = '\n\n'

class RnnRevWS2(RnnWFWS2):

	def tokens2nndata(self, train_text, train_tags=None):
		'''
		@summary: 将输入文本转化为id序列 and reverse the input sequences.
		'''
		# reverse
		train_text = train_text[::-1]
		if train_tags:
			train_tags = train_tags[::-1]

		return super(RnnRevWS2, self).tokens2nndata(train_text, train_tags)

	def segment(self, text, decode=True):

		res = super(RnnRevWS2, self).segment(text, decode, rev=True)

		return res

class RnnFRWS(object):

	def __init__(self, fws, rws):

		self.fws = fws
		self.rws = rws

	def segment(self, text):

		tags1, pm1 = self.fws.segdecode(text, decode=True)
		tags2, pm2 = self.rws.segdecode(text, decode=True, rev=True)

		gps = self.findDiff(tags1, tags2)

		for pair in gps:
			s = pair[0]
			e = pair[1] + 1
			a = tags1[s:e]
			b = tags2[s:e]
			s1, s2 = self.calDiff(pair, tags1, pm1, tags2, pm2)

			if s1 >= s2:	# user the result of forward
				for i in range(s, e):
					tags2[i] = tags1[i]

		return formtext(text, tags2)

	# find the different
	def findDiff(self, tags1, tags2):
		last = -1
		groups = []
		pair = []
		for i in range(len(tags1)):
			if tags1[i] != tags2[i]:
				if i != last+1:
					if len(pair) == 1:
						pair.append(last)
						groups.append(pair)
					pair = [i]
				last = i

		if len(pair) == 1:
			pair.append(last)
			groups.append(pair)

		return groups

	def calDiff(self, pair, tags1, pm1, tags2, pm2):
		"find the surround probs"
		# sum1 = 0.
		# sum2 = 0.
		sidx = min(self.findPre(tags1, pair[0]), self.findPre(tags2, pair[0]))
		eidx = max(self.findSuf(tags1, pair[1]), self.findSuf(tags2, pair[1]))
		# for i in range(sidx, eidx):
			# sum1 += pm1[i][tags1[i]]
			# sum2 += pm2[i][tags2[i]]

		tsum1 = 0.
		tsum2 = 0.
		# sidx = max(findPre(tags1, sidx-1), findPre(tags2, sidx-1))
		# sidx = max(findPre(tags1, sidx-1), findPre(tags2, sidx-1))
		sidx = max(sidx - 2, 0)
		eidx = min(eidx + 2, len(tags1))
		for i in range(sidx, eidx):
			tsum1 += pm1[i][tags1[i]]
			tsum2 += pm2[i][tags2[i]]

		return tsum1, tsum2

	def findPre(self, tags, idx):
		if idx < 0:
			return 0

		if tags[idx] == 2 or tags[idx] == 3:
			while idx >= 0 and tags[idx] != 1:
				idx -= 1

		return max(idx, 0)

	def findSuf(self, tags, idx):
		# idx + 1
		len_t = len(tags)
		if idx >= len_t:
			return len_t

		if tags[idx] == 1 or tags[idx] == 2:
			while idx < len_t and tags[idx] != 3:
				idx += 1

		return min(idx+1, len_t)
