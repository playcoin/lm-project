# -*- coding: utf-8 -*-
'''
Created on 2013-04-12 19:27
@summary: N-gram类
@author: egg
'''
import cPickle
import numpy
import time
from LMBase import LMBase

class Ngram(LMBase):
	'''
	@summary: N-gram类，用于N-gram的训练、预测和评估 
	'''

	def __init__(self, ndict, N=4, ngram_file_path=None):
		'''
		@summary: 构造函数，如果有路径参数则直接读取
		
		@param ndict: 词典
		@param n_size: N-gram的N值
		@param ngram_file_path: 模型序列化文件的路径
		'''

		super(Ngram, self).__init__(ndict)

		self.N = N < 1 and 1 or N
		self.SMOOTH_COEF = 1.
		self.dismap = {}
		self.alphamap = {}
		if ngram_file_path:
			self.loadmodel(ngram_file_path)
		else:
			self.__objinit()

	def __objinit(self):
		'''
		@summary: 初始化成员变量的私有函数。该函数构造N个计数概率对象（cpmap）
			cpmap 是python的dict对象，key是n-1的token序列，value是另一个dict，
			存放第n个token的出现次数和频率( {tid : {count, prop}} )。
		'''

		# 构造N个cpmap
		self.cpmaps = [{} for i in range(self.N)]
		self.k = 10
		self.fofs = [[0 for j in xrange(self.k+2)] for i in range(self.N)]
		self.train_token_size = 0;

	def __traintidseq(self, tidseq):
		'''
		@summary: 输入一个token id 的序列，训练N-gram（统计N-gram的次数）
		
		@param tidseq: token id 序列
		'''

		# 遍历tidseq，每读一个id，就更新能够更新的cpmap
		N = self.N
		n_ran = range(N)
		seq_size = len(tidseq)
		for i_tid in xrange(seq_size):
			# 由小到大检查各个gram
			for i_n in n_ran:
				if i_tid >= i_n:
					self.__updatecount(i_n+1, tidseq[i_tid-i_n:i_tid+1])

		self.train_token_size += seq_size

	def __updatecount(self, n, tids):
		'''
		@summary: 更新某个gram的计数（cpmap中的count）
		
		@param n: 更新的n-gram
		@param tids: 对应的gram序列
		'''
		cpmap = self.cpmaps[n-1]
		# print type(cpmap), n, tids

		ctid = tids[-1]
		key = tuple((len(tids) > 1 and tids[:-1] or [-1]))
		if key in cpmap:
			cpobj = cpmap[key]
			if ctid in cpobj:
				cpobj[ctid][0] += 1
			else:
				cpobj[ctid] = [1, 0.]
		else:
			cpmap[key] = {ctid : [1, 0.]}

	def __updateprop(self):
		'''
		@summary: 更新每个cpmap中的概率值 
		'''

		# 第一层遍历各个n-gram的cpmap
		for cpmap in self.cpmaps:
			# 第二层遍历各个cpmap中的各个gram
			for key, cpobj in cpmap.viewitems():
				# 先求和
				csum = 0.
				for pair in cpobj.viewvalues():
					csum += pair[0]
				# 再计算概率
				for tid, pair in cpobj.viewitems():
					cpobj[tid][1] = pair[0] / csum

	def __calfof(self):
		'''
		@summary: Frequency of frequency
		'''
		# iterate all cpmap of each n-gram
		for i in range(self.N):
			fof = self.fofs[i]
			cpmap = self.cpmaps[i]

			for key, cpobj in cpmap.viewitems():
				for pair in cpobj.viewvalues():
					count = pair[0]
					if count <= self.k + 1:
						fof[count] += 1

					fof[0] += count

		# print self.fofs

	def __updatelambdas(self):
		'''
		@summary: update the lambdas corresponding to each n-gram,
				  count the times n-gram occur, then normalize.
		'''

		# count occur times for each n-gram
		clambdas = [0 for i in range(self.N)]
		# only check the N-gram
		cpmap = self.cpmaps[self.N-1];
		for key, cpobj in cpmap.viewitems():
			for tid, pair in cpobj.viewitems():
				# get N prob of each n-gram
				n_probs = []
				for i in xrange(self.N):
					if i == 0:
						n_probs.append(self.cpmaps[0][(-1,)][tid][1])
					else:
						n_probs.append(self.cpmaps[i][key[-i:]][tid][1])

				# get the index of max value 
				# print n_probs
				i_lambda = numpy.argmax(n_probs)
				clambdas[i_lambda] += 1

		# normalize
		total = numpy.sum(clambdas)

		self.lambdas = [(float(lam) / total) for lam in clambdas]
		# print self.lambdas

	def traintokenseq(self, tokenseq, add_se=True):
		'''
		@summary: 输入一个token 列表，转化为id列表，并进行训练
		
		@param tokenseq: token的文本序列
		@param add_se: 是否添加首尾符号
		'''

		# 借助dict将token串，转为id串
		tidseq = self.tokens2ids(tokenseq, add_se)

		# 训练tid序列
		self.__traintidseq(tidseq)

	def traintext(self, text, seqsplit=None):
		'''
		@summary: 训练一段文本。文本按换行符切分
		
		@param text: 输入文本，默认不包含空白符，
		'''
		
		print "N(%s)-gram train start!!" % self.N
		if(seqsplit):
			lines = text.split(seqsplit)
			[self.traintokenseq(line) for line in lines if line != ""]
		else:
			self.traintokenseq(text)

		print "Count over! Calculate prop!"
		# self.__updateprop()
		self.__calfof()
		# self.__updatelambdas()
		print "N(%s)-gram Train Over!" % self.N

	def testtext(self, text):
		return

	def getcount(self, tids):
		'''
		@summary: Return the count of token seqs
		
		@param tids:
		'''

		length = len(tids)
		if length > self.N:
			return 0
		elif length == 0:
			return self.fofs[0][0]

		pre_tids = tids[:-1]
		n = len(pre_tids)
		ctid = tids[-1]

		cpmap = self.cpmaps[n]
		key = tuple((n > 0 and pre_tids or [-1]))

		if key in cpmap and ctid in cpmap[key]:
			return cpmap[key][ctid][0]
		else:
			return 0

	def cpdiscount(self, count, n):
		'''
		@summary: Return the discount frequency of a n-gram text
		
		@param count: the origin count
		@param n: the current length of text
		@result: 
		'''

		if count > self.k:
			return count
		else:
			fof = self.fofs[n-1]
			return (float(count + 1) * fof[count + 1] / fof[count] - count*float(self.k + 1) * fof[self.k+1] / fof[1] ) / (1 - float(self.k + 1) * fof[self.k+1] / fof[1])

	def probdiscount(self, tids, count=None):
		'''
		@summary: Return the discount probability of a n-gram text
		@result: 
		'''
		count = count or self.getcount(tids)
		tidskey = tuple(tids)
		if tidskey in self.dismap:
			return self.dismap[tidskey]

		if count == 0 and len(tids) == 1:
			propdis = float(self.fofs[0][1]) / self.fofs[0][0]
			self.dismap[tidskey] = propdis
			return propdis

		discount = self.cpdiscount(count, len(tids))

		propdis = float(discount) / self.getcount(tids[:-1])
		self.dismap[tidskey] = propdis
		return propdis

	def alpha(self, tids):
		'''
		@summary: Return the alpha value of the history text 
		
		@param tids: history token ids
		'''
		if self.getcount(tids) < 1:
			return 1.

		tidskey = tuple(tids)
		if tidskey in self.alphamap:
			return self.alphamap[tidskey]

		numerator = 1.
		denominator = 1.

		for i in xrange(self.ndict.size()):
			n_tids = tids + [i]
			count = self.getcount(n_tids)
			if count > 0:
				# print count, n_tids
				numerator -= self.probdiscount(n_tids, count)
				denominator -= self.probdiscount(n_tids[1:])

		# equal judgement for float
		if abs(numerator-0.) < 0.00001 or abs(denominator-0.) < 0.00001:
			self.alphamap[tidskey] = 1.
			return 1.

		# print numerator, denominator
		value = numerator / denominator
		self.alphamap[tidskey] = value
		return value

	def backoff(self, tids):
		'''
		@summary: 计算给定token id序列的概率
		
		@param tids: token id序列
		@result: 序列概率
		'''

		n = len(tids)
		count = self.getcount(tids)

		if count == 0 and n > 1:
			# print tids
			prob = self.alpha(tids[:-1]) * self.backoff(tids[1:])
			return prob

		prob = self.probdiscount(tids, count)
		return prob

	def interpolation(self, tids):
		'''
		@summary: Calculate the iterpolation smooth value
		
		@param tids:
		'''
		# get the last N tokens
		tids = tids[-self.N:]
		n = len(tids)
		ctid = tids[-1]

		# get the n-gram probablities
		n_probs = []
		while n > 0:
			cpmap = self.cpmaps[n-1]
			key = tuple((n > 1 and tids[:-1] or [-1]))
			prob = 0.0
			# print key, cpmap[key]
			if key in cpmap and ctid in cpmap[key]:
				prob = cpmap[key][ctid][1]
				
			n -= 1
			tids = tids[-n:]
			n_probs.append(prob)

		# reverse the probs
		n_probs.reverse()
		# print n_probs

		# cal prob by for loop
		f_prob = 0.0
		for i in range(len(n_probs)):
			f_prob += n_probs[i] * self.lambdas[i]
			# print n_probs[i], self.lambdas[i]

		# 如果 prob 还是 0，那么就给个最很小的值
		f_prob = f_prob == 0. and (self.SMOOTH_COEF / self.train_token_size) or f_prob

		return f_prob


	def predict(self, text):

		# text length should be large than 0
		if(len(text) < 1):
			return None

		# last N-1 token will be enough
		his_text = text[-self.N+1:]
		his_tids = [self.ndict[token] for token in his_text]
		# check whether cpmaps has the tuple
		len_text = len(his_tids)
		# the max prob and the corresponding token id
		p_max, tid_max = (0, self.ndict.getunknownindex())
		
		while len_text > 0:
			# get the cpmap 
			cpmap = self.cpmaps[len_text]
			key = tuple(his_tids)
			if key in cpmap:
				# iter the cpobj to get the max probability
				cpobj = cpmap[key]
				
				# 再计算概率
				for tid, pair in cpobj.viewitems():
					if pair[1] > p_max:
						p_max = pair[1]
						tid_max = tid

				break;

			else:
				his_tids = his_tids[-len_text+1:]
				len_text -= 1

		return tid_max, p_max

	def rank(self, tids):
		'''
		@summary: cal the rank for one tuple
		
		@param tids:
		'''

		probs = []
		prefix = tids[:-1]
		dict_size = self.ndict.size()
		for i in range(dict_size):
			prob = self.backoff(prefix + [i])
			probs.append(prob)


		# print probs
		probs = numpy.asarray(probs)
		probs.sort()
		prob = self.backoff(tids)

		return dict_size - probs.searchsorted(prob)

	def ranks(self, text):
		tidseq = self.tokens2ids(text)

		rank_list = []
		len_seq = len(tidseq)
		stime = time.clock()
		for i in xrange(len_seq):
			sub_seq = []
			if i < self.N:
				sub_seq = tidseq[:i+1]
			else:
				sub_seq = tidseq[i-self.N+1:i+1]

			rank_list.append(self.rank(sub_seq))
			if (i+1) % 1000 == 0:
				print "%s/%s, time: %ss" % (i+1, len_seq, time.clock() - stime)
			# print rank_list

		return rank_list

	def topN(self, tids, N=10):
		'''
		@summary: Return the top N predict char of the history tids
		'''
		probs = []
		prefix = tids
		dict_size = self.ndict.size()
		for i in range(dict_size):
			prob = self.backoff(prefix + [i])
			probs.append(prob)

		sort_probs = numpy.asarray(probs)
		sort_probs.sort()
		top_probs = sort_probs[-N:][::-1]

		top_tokens = [probs.index(x) for x in top_probs]
		return top_tokens, top_probs


	def likelihood(self, text, add_se=False, smoothfuncname = "backoff"):

		# turn text sequence to token id sequence
		tidseq = self.tokens2ids(text)

		# choose smooth function
		smoothfunc = self.backoff
		if smoothfuncname == 'interpolation':
			smoothfunc = self.interpolation

		# cal the log probability
		# cal cross-entropy first, use the equation:
		# 	H(W) = - 1 / N * (\sum P(w_1 w_2 ... w_N))
		prop = []
		len_seq = len(tidseq)
		for i in xrange(len_seq):
			sub_seq = []
			if i < self.N:
				sub_seq = tidseq[:i+1]
			else:
				sub_seq = tidseq[i-self.N+1:i+1]

			prop.append(smoothfunc(sub_seq))

		return prop

	def crossentropy(self, text, add_se=False):

		# turn text sequence to token id sequence
		tidseq = self.tokens2ids(text)

		# cal the log probability
		# cal cross-entropy first, use the equation:
		# 	H(W) = - 1 / N * (\sum P(w_1 w_2 ... w_N))
		log_prob = []
		len_seq = len(tidseq)
		for i in xrange(len_seq):
			sub_seq = []
			if i < self.N:
				sub_seq = tidseq[:i+1]
			else:
				sub_seq = tidseq[i-self.N+1:i+1]

			prob = self.backoff(sub_seq)
			log_prob.append(numpy.log(prob))

		crossentropy = - numpy.mean(log_prob)

		return crossentropy, log_prob

	def savemodel(self, filepath="./data/ngram.model.obj"):

		backupfile = open(filepath, 'w')
		cPickle.dump((self.cpmaps, self.train_token_size, self.lambdas), backupfile)
		backupfile.close()

	def loadmodel(self, filepath="./data/ngram.model.obj"):

		backupfile = open(filepath)
		self.cpmaps, self.train_token_size, self.lambdas = cPickle.load(backupfile)
		backupfile.close()
		print "Load model complete!"






