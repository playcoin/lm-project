# -*- coding: utf-8 -*-
'''
Created on 2013-04-12 19:27
@summary: N-gram类
@author: egg
'''
import cPickle
import numpy

class Ngram(object):
	'''
	@summary: N-gram类，用于N-gram的训练、预测和评估 
	'''

	def __init__(self, nlpdict, N=4, ngram_file_path=None):
		'''
		@summary: 构造函数，如果有路径参数则直接读取
		
		@param nlpdict: 词典
		@param n_size: N-gram的N值
		@param ngram_file_path: 模型序列化文件的路径
		'''

		self.ndict = nlpdict
		self.N = N < 1 and 1 or N
		self.SMOOTH_COEF = 0.005
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
		print self.lambdas

	def traintokenseq(self, tokenseq, add_se=True):
		'''
		@summary: 输入一个token 列表，转化为id列表，并进行训练
		
		@param tokenseq: token的文本序列
		@param add_se: 是否添加首尾符号
		'''

		# 借助dict将token串，转为id串
		# print tokenseq
		tidseq = add_se and [self.ndict.getstartindex()] or []
		tidseq.extend([self.ndict[token] for token in tokenseq])
		add_se and tidseq.extend([self.ndict.getendindex()])
		# print tidseq

		# 训练tid序列
		self.__traintidseq(tidseq)

	def traintext(self, text, seqsplit=None):
		'''
		@summary: 训练一段文本。文本按换行符切分
		
		@param text: 输入文本，默认不包含空白符，
		'''
		
		print "N-gram train start!!"
		if(seqsplit):
			lines = text.split(seqsplit)
			[self.traintokenseq(line) for line in lines if line != ""]
		else:
			self.traintokenseq(text)

		print "Count over! Calculate prop!"
		self.__updateprop()
		self.__updatelambdas()
		print "N-gram Train Over!"

	def backoff(self, tids):
		'''
		@summary: 计算给定token id序列的概率
		
		@param tids: token id序列
		@result: 序列概率
		'''

		# 只取序列的最后N个
		tids = tids[-self.N:]
		n = len(tids)
		ctid = tids[-1]
		prob = 0.
		# print tids
		while n > 0 and prob == 0.:
			cpmap = self.cpmaps[n-1]
			key = tuple((n > 1 and tids[:-1] or [-1]))
			# print key, cpmap[key]
			if key in cpmap and ctid in cpmap[key]:
				prob = cpmap[key][ctid][1]
			else:
				n -= 1
				tids = tids[-n:]

		# 如果 prob 还是 0，那么就给个最很小的值
		prob = prob == 0. and (self.SMOOTH_COEF / self.train_token_size) or prob

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
		'''
		@summary: 给定一段文本，预测下一个token
		
		@param text:
		@result: 
		'''
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

	def likelihood(self, text, add_se=False, smoothfuncname = "interpolation"):
		'''
		@summary: Return the likelihood of the token sequence
		
		@param text:
		@param add_se:
		@result: 
		'''
		# turn text sequence to token id sequence
		tidseq = add_se and [self.ndict.getstartindex()] or []
		tidseq.extend([self.ndict[token] for token in text])
		add_se and tidseq.extend([self.ndict.getendindex()])

		# choose smooth function
		smoothfunc = self.interpolation
		if smoothfuncname == 'backoff':
			smoothfunc = self.backoff

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
		'''
		@summary: Return the cross-entropy of the text
		
		@param text:
		@result: cross entropy
		'''
		# turn text sequence to token id sequence
		tidseq = add_se and [self.ndict.getstartindex()] or []
		tidseq.extend([self.ndict[token] for token in text])
		add_se and tidseq.extend([self.ndict.getendindex()])

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

			log_prob.append(numpy.log(self.interpolation(sub_seq)))

		crossentropy = - numpy.sum(log_prob) / len_seq

		return crossentropy, log_prob

	def perplexity(self, text, add_se=False):
		'''
		@summary: Return the perplexity of the text
		
		@result: perplexity
		'''
		return numpy.exp2(self.crossentropy(text, add_se)[0])

	def savemodel(self, filepath="./data/ngram.model.obj"):
		'''
		@summary: Save model to file
		
		@param filepath: back up file path
		'''

		backupfile = open(filepath, 'w')
		cPickle.dump((self.cpmaps, self.train_token_size, self.lambdas), backupfile)
		backupfile.close()

	def loadmodel(self, filepath="./data/ngram.model.obj"):
		'''
		@summary: Load model from file
		
		@param filepath:
		'''

		backupfile = open(filepath)
		self.cpmaps, self.train_token_size, self.lambdas = cPickle.load(backupfile)
		backupfile.close()
		print "Load model complete!"






