# -*- coding: utf-8 -*-
'''
Created on 2013-04-12 17:50
@summary: 字典类
@author: egg
'''
import cPickle
import re
import numpy


UNKNOWN_STR = "$_UNKNOWN_$"
SDIGIT_STR = "$_SDIGIT_$"
BDIGIT_STR = "$_BDIGIT_$"
ENCHAR_STR = "$_ENCHAR_$"

SDIGIT_SET = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
BDIGIT_SET = {u"０", u"１", u"２", u"３", u"４", u"５", u"６", u"７", u"８", u"９"}
ENCHAR_SET = {
	"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",\
	"k", "l", "m", "n", "o", "p", "q", "r", "s", "t",\
	"u", "v", "w", "x", "y", "z",\
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J",\
	"K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",\
	"U", "V", "W", "X", "Y", "Z",\
	u"ａ", u"ｂ", u"ｃ", u"ｄ", u"ｅ", u"ｆ", u"ｇ", u"ｈ", u"ｉ", u"ｊ",\
	u"ｋ", u"ｌ", u"ｍ", u"ｎ", u"ｏ", u"ｐ", u"ｑ", u"ｒ", u"ｓ", u"ｔ",\
	u"ｕ", u"ｖ", u"ｗ", u"ｘ", u"ｙ", u"ｚ",\
	u"Ａ", u"Ｂ", u"Ｃ", u"Ｄ", u"Ｅ", u"Ｆ", u"Ｇ", u"Ｈ", u"Ｉ", u"Ｊ",\
	u"Ｋ", u"Ｌ", u"Ｍ", u"Ｎ", u"Ｏ", u"Ｐ", u"Ｑ", u"Ｒ", u"Ｓ", u"Ｔ",\
	u"Ｕ", u"Ｖ", u"Ｗ", u"Ｘ", u"Ｙ", u"Ｚ"
}


class NlpDict(object):
	'''
	@summary: 字典类，包括一些基本的字典操作，存放token的ID
	'''

	def __init__(self, dict_file_path=None, comb=False, comben=True):
		'''
		@summary: 构造函数，允许接收一个字符串作为词典备份路径作为参数。如果该参数不为空，则需要加载

		@param dict_file_path:
		@param comb: 是否合并数字
		'''
		self.comb = comb
		self.comben = comben

		if dict_file_path:
			# 调用load方法，并赋值给self
			self.loadNlpDict(dict_file_path)
		else:
			self.__dictinit()

	def __dictinit(self):
		'''
		@summary: 初始化词典（私有方法）。如果再次调用，会清空词典
		'''
		self.ndict = {}	# token to ID
		self.ndict_inv = [] # ID to token
		# special token, '$_START_$' and '$_END_$' is useless in training now.
		self.ndict_inv.append(UNKNOWN_STR)
		# self.ndict_inv.append(UNKNOWN_STR+'1')
		# self.ndict_inv.append(UNKNOWN_STR+'2')
		if self.comb:
			self.ndict_inv.extend([SDIGIT_STR, BDIGIT_STR])
		for elm in self.ndict_inv:
			self.ndict[elm] = len(self.ndict)

	def __combdigit(self, char):
		if char in SDIGIT_SET:
			return SDIGIT_STR
		elif char in BDIGIT_SET:
			return BDIGIT_STR
		elif self.comben and (char in ENCHAR_SET):
			return ENCHAR_STR
		else:
			return char

	def __len__(self):
		'''
		@summary: 专用方法，返回词典大小
		'''

		return len(self.ndict)

	def __getitem__(self, token):
		'''
		@summary: 专用方法，通过token的文本，获取token的ID
		
		@param token: token 文本
		'''

		return self.getindex(token)

	def gettoken(self, index):
		'''
		@summary: 通过token ID 获取token的文本
		
		@param index: token ID
		'''

		if index >= 0 and index <= len(self.ndict_inv):
			return self.ndict_inv[index]
		else:
			return UNKNOWN_STR

	def getindex(self, token):
		'''
		@summary: 通过token的文本，获取token的ID 
		
		@param token: token 文本
		'''
		if self.comb: # 检查是否要合并
			token = self.__combdigit(token)

		if token in self.ndict:
			return self.ndict[token]
		else:
			return self.ndict[UNKNOWN_STR]

	def addtoken(self, token):
		'''
		@summary: 添加一个元素到字典中

		@param token:
		'''

		if token not in self.ndict:
			self.ndict_inv.append(token)
			self.ndict[token] = len(self.ndict)

	def size(self):
		'''
		@summary: 获取词典大小
		'''

		return len(self.ndict)

	def getunknownindex(self):
		'''
		@summary: 获取未知符的编号
		'''

		return self.getindex(UNKNOWN_STR)

	def buildfromfile(self, text_file_path, white_space=False, freq_thres=0):
		'''
		@summary: 读取文本文件，并通过文本填充词典

		@param text_file_path:
		@param white_space:
		@param freq_thres:
		'''
		# 读取文本
		corpus = file(text_file_path)
		text = corpus.read()
		corpus.close()
		# 转化为unicode
		text = unicode(text, 'utf-8')

		self.buildfromtext(text, white_space, freq_thres)

	def buildfromtext(self, text, white_space=False, freq_thres=0):
		'''
		@summary: 直接通过文本构造词典
		
		@param text: 文本, 只支持unicode
		@param white_space: 是否保留空白符
		@param freq_thres: 是否清除低频词
		'''
		# 重新初始化内部的dict对象
		self.__dictinit()
		# 处理空白字符，如果需要的话，就留着
		if not white_space:
			text = re.sub(r'\t| ', '', text)
		# 清理低频词
		if freq_thres > 0:
			# 计算词频
			ndict = {}
			for char in text:
				if self.comb:	# 检查是否要合并数字
					char = self.__combdigit(char)

				if char in ndict:
					ndict[char] += 1
				else:
					ndict[char] = 1
			# 再删掉词少的
			for char, count in ndict.items():
				if count <= freq_thres:
					del ndict[char]
			# 添加到词典
			for char in ndict:
				self.addtoken(char)
		else:	# 阈值为0的话，直接添加即可
			for char in text:
				# 检查是否要合并数字
				if self.comb:
					char = self.__combdigit(char)
				self.addtoken(char)


	def storeNlpDict(self, dict_file_path):
		'''
		@summary: 利用cPickle将NlpDict对象序列化保存
		
		@param dict_file_path: 保存路径
		'''

		dict_file = open(dict_file_path, 'wb')

		cPickle.dump((self.ndict, self.ndict_inv), dict_file)
		dict_file.close()

	def loadNlpDict(self, dict_file_path):
		'''
		@summary: 利用cPickle读取NlpDict对象
		
		@param dict_file_path: 读取路径
		@result: NlpDict对象
		'''
		dict_file = open(dict_file_path)

		self.ndict, self.ndict_inv = cPickle.load(dict_file)
		dict_file.close()

	def transEmbedding(self, embedding_file_path, hv_file_path):
		'''
		@summary: 读取embedding的结果
		
		@param embedding_file_path:
		'''
		emfile = open(embedding_file_path)
		text = unicode(emfile.read(), 'utf-8')
		emfile.close()

		# 逐行读取
		lines = text.split('\n')
		ev_map = {}
		for i in xrange(len(lines)):
			units = lines[i].split(', ')
			units[0] = (units[0] == "$_SENT_$") and '\n' or units[0]		# for sentence end

			if units[0] == '$EN$':	# for English characters
				ev_values = [float(x) for x in units[1:-1]]
				import string
				for char in string.letters:
					if char in self.ndict:
						ev_map[char] = ev_values

				continue	# skip to next token

			if len(units) > 1 and units[0] in self.ndict:
				ev_map[units[0]] = [float(x) for x in units[1:-1]]

		emvalues = []
		for i in xrange(self.size()):
			char = self.gettoken(i)
			if char in ev_map:
				ev = ev_map[char]
			else:
				ev = ev_map['$_UNKNOWN_$']

			emvalues.append(ev)

		emvalues = numpy.asarray(emvalues, dtype="float32")
		print emvalues.shape

		hv_file = open(hv_file_path, 'wb')

		cPickle.dump(emvalues, hv_file)
		hv_file.close()