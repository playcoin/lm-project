# -*- coding: utf-8 -*-
'''
Created on 2013-12-23 13:39
@summary: 词性标注器
@author: Playcoin
'''

import numpy
import cPickle
import time
import re

import theano
from pyws import RnnWFWS
from tagutil import tagmap, tagsize, formtext

class RnnPOS(RnnWFWS):
	'''
	@summary: 词性标注器
	'''

	def __init__(self, *args, **kwargs):
		super(RnnPOS, self).__init__(*args, **kwargs)

		self.out_size = tagsize
		self.septag = tagmap["w_b"]
		self.tagsize = tagsize

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

		mat_size = len(mat_in)
		mat_in = theano.shared(numpy.asarray(mat_in, dtype="int32"), borrow=True)

		if train_tags:
			# 词性标注的tag编号是大于10的，所以文本中用空格隔开
			repstr = "  %d  %d  " % (self.septag, self.septag)	# 将回车符转变一下
			train_tags = train_tags.strip().replace("\n", repstr)
			# 切分并变为数字
			train_tag_nums = re.split(r"\s+", train_tags)
			tags = [int(tag) for tag in train_tag_nums[:mat_size]]

			vec_out = theano.shared(numpy.asarray(tags, dtype="int32"), borrow=True)
	
			return mat_in.get_value(borrow=True), vec_out.get_value(borrow=True)
		else:
			return mat_in.get_value(borrow=True)

	def acumPrior(self, train_tags):
		'''
		@summary: Acumulate the prior probabilities of tag sequences
		'''
		priorMatrix = numpy.zeros((self.tagsize, self.tagsize), dtype=theano.config.floatX)
		pi = numpy.zeros((self.tagsize, ), dtype=theano.config.floatX)
		num_total = 0
		# process line
		taglines = train_tags.strip().split('\n')
		for line in taglines:
			assert line.strip() != ""
			tags = [int(x) for x in re.split(r'\s+', line.strip())]
			pi[tags[0]] += 1
			for i in range(1, len(tags)):
				priorMatrix[tags[i-1]][tags[i]] += 1
			num_total += len(tags) - 1

		self.pm = priorMatrix / num_total
		self.logpm = numpy.log(self.pm)
		self.pi = pi / len(taglines)
		self.logpi = numpy.log(self.pi)

	def decode(self, text, rev=False):
		'''
		@summary: 解码 BMES tag, S:0, B:1, M:2, E:3
		'''

		self.initRnn()
		data_input = self.tokens2nndata(text)

		if type(data_input) == tuple:
			prob_matrix = numpy.log(self.rnn_prob_matrix(*data_input))
		else:
			prob_matrix = numpy.log(self.rnn_prob_matrix(data_input))

		if rev:
			prob_matrix = numpy.flipud(prob_matrix)

		# 解码
		old_pb = prob_matrix.copy()
		tm = numpy.zeros((len(prob_matrix), self.tagsize), dtype="int32")
		# return self.rnn_pred(data_input), old_pb

		# print prob_matrix
		# print self.pi
		for i in range(0, self.tagsize):
			prob_matrix[0][i] += self.logpi[i]

		for i in xrange(1, len(prob_matrix)):
			for k in range(0, self.tagsize):
				max_idx = 0
				max_pb = -999999.
				for j in range(0, self.tagsize):
					# pb = prob_matrix[i-1][j] + self.logpm[j][k]
					# if pb > max_pb:
					# 	max_pb = pb
					# 	max_idx = j
					if self.pm[j][k] > 0.:
						pb = prob_matrix[i-1][j]
						if pb > max_pb:
							max_pb = pb
							max_idx = j


				prob_matrix[i][k] += max_pb
				tm[i][k] = max_idx

		last = prob_matrix[-1].argmax()
		tags = [last]
		for i in xrange(len(prob_matrix) - 1, 0, -1):
			last = tm[i][last]
			tags.append(last)
		# reverse
		tags.reverse()

		return tags, old_pb

	def segment(self, text, rev=False):
		'''
		@summary: 先解码，在格式化
		'''
		tags, prob_matrix = self.decode(text, rev)

		return formtext(text, tags)


class RnnRevPOS(RnnPOS):

	def tokens2nndata(self, train_text, train_tags=None):
		'''
		@summary: 将输入文本转化为id序列 and reverse the input sequences.
		'''
		# reverse
		train_text = train_text[::-1]
		train_text = self.train_preffix + train_text.strip().replace("\n", "\n\n") + self.train_suffix
		tids = [self.ndict[token] for token in train_text]

		mat_in = []
		for i in xrange(0, len(tids) - self.ext_emb):
			mat_in.append(tids[i:i+self.ext_emb+1])

		mat_size = len(mat_in)
		mat_in = theano.shared(numpy.asarray(mat_in, dtype="int32"), borrow=True)

		if train_tags:
			# 词性标注的tag编号是大于10的，所以文本中用空格隔开
			repstr = "  %d  %d  " % (self.septag, self.septag)	# 将回车符转变一下
			train_tags = train_tags.strip().replace("\n", repstr)
			# 切分并变为数字
			train_tag_nums = re.split(r"\s+", train_tags)
			tags = [int(tag) for tag in train_tag_nums]
			tags = tags[::-1][:mat_size]

			vec_out = theano.shared(numpy.asarray(tags, dtype="int32"), borrow=True)
	
			return mat_in.get_value(borrow=True), vec_out.get_value(borrow=True)
		else:
			return mat_in.get_value(borrow=True)