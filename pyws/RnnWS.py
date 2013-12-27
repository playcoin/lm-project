# -*- coding: utf-8 -*-
'''
Created on 2013-11-28 12:27
@summary: Word Segmentation Module by RNN
@author: egg
'''
import math
import numpy
import cPickle
import time

import theano
import theano.tensor as T
from theano.tensor.sharedvar import TensorSharedVariable
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
from dltools.rnnemb import RNNEMB
from tagutil import formtext

class RnnWS(object):

	def __init__(self, ndict, n_emb, n_hidden, lr, batch_size, 
		l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5, emb_dr_rate = 0.,
		emb_file_path = None, backup_file_path=None):

		self.ndict = ndict

		if backup_file_path is None:
			self.n_hidden = n_hidden
			self.lr = lr
			self.batch_size = batch_size
			self.truncate_step = truncate_step
			self.rnnparams = None
			self.embvalues = None
		else:
			self.loadmodel(backup_file_path)

		self.rnn = None
		self.dr_rate = dr_rate
		self.emb_dr_rate = emb_dr_rate
		self.train_emb = train_emb
		self.in_size = ndict.size()
		self.out_size = 4
		self.n_emb = n_emb
		self.l2_reg = l2_reg

		if emb_file_path:
			f = open(emb_file_path)
			self.embvalues = cPickle.load(f)
			f.close()
			self.embvalues = self.embvalues.astype(theano.config.floatX)
			
	def initRnn(self, no_train=False):
		'''
		@summary: Initiate RNNEMB model 
		'''
		if self.rnn is not None:
			return

		print "%s init start!" % self.__class__.__name__
		u = T.ivector('u')
		y = T.ivector('y')
		l = T.imatrix('l')
		h_init = T.matrix('h_init')

		rng = numpy.random.RandomState(213234)
		rnn = RNNEMB(rng, 
				self.in_size,
				self.n_emb, 
				self.n_hidden, 
				self.out_size,
				self.batch_size,
				self.lr,
				dr_rate = self.dr_rate,
				emb_dr_rate = self.emb_dr_rate,
				params = self.rnnparams,
				embeddings = self.embvalues
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
		# 将训练文本再预处理一下, 单个回车符编程两个，回车符的标签为0
		train_text = train_text.replace("\n", "\n\n") + "\n"
		tids = [self.ndict[token] for token in train_text]
		vec_in = theano.shared(numpy.asarray(tids, dtype="int32"), borrow=True)

		if train_tags:
			train_tags = "0" + train_tags.replace("\n", "00")
			tags = [int(tag) for tag in train_tags]
			vec_out = theano.shared(numpy.asarray(tags, dtype="int32"), borrow=True)

			return vec_in.get_value(borrow=True), vec_out.get_value(borrow=True)

		return vec_in.get_value(borrow=True)

	def reshape(self, dataset, data_size):
		'''
		@summary: 将训练数据按batch_size进行reshape
		'''
		out_dataset = []
		for x in dataset:
			x = x[:data_size]
			s_x = x.reshape(self.batch_size, x.shape[0] / self.batch_size).T
			out_dataset.append(theano.shared(s_x, borrow=True))
		return out_dataset

	def traintext(self, train_text, train_tags, test_text, test_tags, 
			sen_slice_length=4, epoch=200, lr_coef = -1., 
			DEBUG = False, SAVE=False, SINDEX=1, r_init=False):

		self.initRnn()

		seq_size = len(train_text)

		# variables for slice sentence
		sentence_length = sen_slice_length * self.truncate_step
		half_sen_length = sentence_length / 2
		data_slice_size = sentence_length * self.batch_size
		data_size = seq_size / data_slice_size * data_slice_size
		# total sentence length after reshape
		total_sent_len = data_size / self.batch_size
		print "Actural training data size: %s, with total sentence length: %s" % (data_size, total_sent_len)

		dataset = self.tokens2nndata(train_text, train_tags)
		tbptt_args = self.reshape(dataset, data_size)
		tbptt_args.extend([self.truncate_step, self.train_emb, self.l2_reg])
		self.rnn.build_tbptt(*tbptt_args)
		print "Compile Truncate-BPTT Algorithm complete!"

		# for test
		test_dataset = self.tokens2nndata(test_text, test_tags)

		s_time = time.clock()
		for i in xrange(epoch):

			for j in xrange(0, total_sent_len - half_sen_length, half_sen_length):
				# reshape to matrix: [SEQ][BATCH][FRAME]
				self.rnn.train_tbptt(j, j+sentence_length)

			if DEBUG:
				err = self.test_model(*test_dataset)
				e_time = time.clock()
				print "Error rate in epoch %s, is %.5f. Training time so far is: %.2fm" % ( i+SINDEX, err, (e_time-s_time) / 60.)

			if SAVE:
				class_name = self.__class__.__name__
				self.savemodel("./data/%s/%s.model.epoch%s.n_hidden%s.ssl%s.truncstep%s.dr%.1f.embsize%s.in_size%s.r%s.obj" % (class_name, class_name, i+SINDEX, self.n_hidden, sen_slice_length, self.truncate_step, self.dr_rate, self.n_emb, self.in_size, r_init))

			if lr_coef > 0:
				# update learning_rate
				lr = self.rnn.lr.get_value() * lr_coef
				self.rnn.lr.set_value(numpy.array(lr, dtype=theano.config.floatX))

		e_time = time.clock()
		print "%s train over!! The total training time is %.2fm." % (self.__class__.__name__, (e_time - s_time) / 60.) 

	def testtext(self, test_text, test_tags, show_result = True):
		'''
		@summary: 测试错误率
		'''
		self.initRnn()

		test_dataset = self.tokens2nndata(test_text, test_tags)
		err = self.test_model(*test_dataset)
		print "Error rate %.5f" % err

	def segment(self, text, decode=True, rev=False):
		'''
		@summary: 先解码，在格式化
		'''
		tags, prob_matrix = self.segdecode(text, decode, rev)

		return formtext(text, tags)

	def segdecode(self, text, decode=True, rev=False):
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

		if not decode:
			return prob_matrix.argmax(1), prob_matrix

		# 解码
		tags = []
		old_pb = prob_matrix.copy()
		# 第一个只可能是 S, B
		prob_matrix[0][2] = -999999.
		prob_matrix[0][3] = -999999.
		for i in xrange(1, len(prob_matrix)):
			prob_matrix[i][0] += max(prob_matrix[i-1][0], prob_matrix[i-1][3])
			prob_matrix[i][1] += max(prob_matrix[i-1][0], prob_matrix[i-1][3])
			prob_matrix[i][2] += max(prob_matrix[i-1][1], prob_matrix[i-1][2])
			prob_matrix[i][3] += max(prob_matrix[i-1][1], prob_matrix[i-1][2])

		c0=0
		c1=3	#two choice
		for i in xrange(len(prob_matrix)-1, -1, -1):
			if(prob_matrix[i][c0] < prob_matrix[i][c1]):
				c0 = c1

			tags.append(c0)
			if c0 == 0:
				c1 = 3 
			elif c0 == 1:
				c0 = 0
				c1 = 3
			elif c0 == 2:
				c1 = 1
			else:
				c0 = 1
				c1 = 2

		tags.reverse()

		return tags, old_pb

	def savemodel(self, filepath):
		backupfile = open(filepath, 'wb')
		# save np data, not the theano shared variable, for different cuda version
		rnnparams = []
		for param in self.rnnparams:
			if type(param) == TensorSharedVariable or type(param) == CudaNdarraySharedVariable:
				param = param.get_value()

			rnnparams.append(param)

		dumpdata = [self.batch_size, self.n_hidden, self.lr, self.truncate_step, rnnparams]
		if self.rnn.C:
			dumpdata.append(self.rnn.C.get_value())

		cPickle.dump(dumpdata, backupfile)
		backupfile.close()
		print "Save model complete! Filepath:", filepath

	def loadmodel(self, filepath):
		try:
			backupfile = open(filepath, 'rb')
			dumpdata = cPickle.load(backupfile)
		except:
			backupfile = open(filepath, 'r')
			dumpdata = cPickle.load(backupfile)

		self.batch_size, self.n_hidden, self.lr, self.truncate_step, rnnparams = dumpdata[:5]
		if len(dumpdata) > 5:
			self.embvalues = dumpdata[5]
		else:
			self.embvalues = None

		self.rnnparams = []
		for param in rnnparams:
			if type(param) != TensorSharedVariable and type(param) != CudaNdarraySharedVariable:
				param = theano.shared(value=param, borrow=True)

			self.rnnparams.append(param)

		backupfile.close()
		print "Load model complete! Filepath:", filepath

	def dumpembedding(self, filepath):
		embfile = open(filepath, 'wb')
		cPickle.dump(self.embvalues, embfile)
		embfile.close()
		print "Dump embeddings complete! Filepath:", filepath
