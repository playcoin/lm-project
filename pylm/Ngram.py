# -*- coding: utf-8 -*-
'''
Created on 2013-04-12 19:27
@summary: N-gram类
@author: egg
'''

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
		if ngram_file_path:
			pass
		else:
			self.__graminit()

	def __graminit(self):
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

	def traintokenseq(self, tokenseq, add_se=True):
		'''
		@summary: 输入一个token 列表，转化为id列表，并进行训练
		
		@param tokenseq: token的文本序列
		@param add_se: 是否添加收尾符号
		'''

		# 借助dict将token串，转为id串
		# print tokenseq
		tidseq = add_se and [self.ndict.getstartindex()] or []
		tidseq.extend([self.ndict[token] for token in tokenseq])
		add_se and tidseq.extend([self.ndict.getendindex()])
		# print tidseq

		# 训练tid序列
		self.__traintidseq(tidseq)

	def traintext(self, text):
		'''
		@summary: 训练一段文本。文本按换行符切分
		
		@param text: 输入文本，默认不包含空白符，
		'''

		lines = text.split('\n')
		print "train start!!"
		[self.traintokenseq(line) for line in lines if line != ""]

		print "train over! Calculate prop!"
		self.__updateprop()

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
		prop = 0.
		# print tids
		while n > 0 and prop == 0.:
			cpmap = self.cpmaps[n-1]
			key = tuple((n > 1 and tids[:-1] or [-1]))
			# print key, cpmap[key]
			if key in cpmap and ctid in cpmap[key]:
				prop = cpmap[key][ctid][1]
			else:
				n -= 1
				tids = tids[-n:]

		# 如果 prop 还是 0，那么就给个最很小的值
		prop = prop == 0. and (0.5 / self.train_token_size) or prop

		return prop

	def predict(self, text):
		'''
		@summary: 给定一段文本，预测下一个token
		
		@param text:
		@result: 
		'''
		pass