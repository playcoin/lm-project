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
from pypos import RnnPOS
from netagutil import tagmap, tagsize

class RnnNER(RnnPOS):
	'''
	@summary: 词性标注器
	'''

	def __init__(self, *args, **kwargs):
		super(RnnNER, self).__init__(*args, **kwargs)

		self.out_size = tagsize

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
			repstr = "  %d  %d  " % (tagmap["o"], tagmap["o"])	# 将回车符转变一下
			train_tags = train_tags.strip().replace("\n", repstr)
			# 切分并变为数字
			train_tag_nums = re.split(r"\s+", train_tags)
			tags = [int(tag) for tag in train_tag_nums[:mat_size]]

			vec_out = theano.shared(numpy.asarray(tags, dtype="int32"), borrow=True)
	
			return mat_in.get_value(borrow=True), vec_out.get_value(borrow=True)
		else:
			return mat_in.get_value(borrow=True)