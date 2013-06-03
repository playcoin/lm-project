# -*- coding: utf-8 -*-
'''
Created on 2013-05-15 10:30
@summary: Mlp version for N-gram
@author: egg
'''

import math
import numpy
import time
import theano
import theano.tensor as T
import cPickle
from dltools.mlp import MLP
from theano import sandbox, Out
from LMBase import LMBase


class MlpNgram(LMBase):
	'''
	@summary: Mlp version for N-gram
	'''

	def __init__(self, ndict, N = 5, n_in=30, n_hidden=40, lr=0.05, l1_reg = 0.00, l2_reg=0.0001, batch_size=40, hvalue_file="./data/MlpBigram.hiddens.obj", backup_file_path=None):
		'''
		@summary: Construct function, set some parameters.
		
		@param ndict:
		@param N:
		@param n_in:
		@param n_hidden:
		@param lr:
		@param l1_reg:
		@param l2_reg:
		@param batch_size:
		@param hvalue_file:
		@param backup_file_path:
		'''

		super(MlpNgram, self).__init__(ndict)

		if backup_file_path is None:
			self.N = N
			self.n_in = n_in
			self.n_hidden = n_hidden
			self.lr = lr
			self.l1_reg = l1_reg
			self.l2_reg = l2_reg
			self.batch_size = batch_size
			self.mlpparams = [None, None, None, None]
		else:
			self.loadmodel(backup_file_path)

		# mlp obj
		self.mlp = None
		self.hvalue_file = hvalue_file
		self.__loadhvalues()

	def __setTrainData(self, train_data):
		'''
		@summary: Set the trainging data, data should be Theano.SharedVariable for GPU.
		'''
		self.train_data = train_data[0]
		self.label_data = train_data[1]

	def __initMlp(self, no_train=False):
		'''
		@summary: Initiate mlp, build theano function of trainging and test. 
		
		@param no_train: whether compile the trainging function
		'''

		if self.mlp is not None:
			return

		print "Init theano symbol expressions!"

		index = T.lscalar()
		x = T.matrix('x')  
		y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

		rng = numpy.random.RandomState(4321)

		# construct the MLP class
		self.mlp = MLP(rng=rng, input=x, n_in=self.n_in * (self.N - 1),
						 n_hidden=self.n_hidden, n_out=self.ndict.size(),
						 hW=self.mlpparams[0], hb=self.mlpparams[1], oW=self.mlpparams[2], ob=self.mlpparams[3])

		self.mlpparams = self.mlp.params

		classifier = self.mlp

		probs = classifier.logRegressionLayer.p_y_given_x[T.arange(y.shape[0]), y]
		self.mlp_prob = theano.function(inputs=[x, y], outputs=probs)
		self.mlp_probs = theano.function(inputs=[x], outputs=classifier.logRegressionLayer.p_y_given_x[-1])
		print "Compile likelihood function complete!"

		y_sort_matrix = T.sort(classifier.logRegressionLayer.p_y_given_x, axis=1)
		self.mlp_sort = theano.function(inputs=[x, y], outputs=[y_sort_matrix, probs])
		print "Compile argsort function complete!"

		y_pred = classifier.logRegressionLayer.y_pred
		self.mlp_predict = theano.function(inputs=[x], outputs=y_pred[-1])
		print "Compile predict function complete!"

		# test model function
		self.test_model = theano.function(inputs=[x, y], outputs=classifier.errors(y))
		print "Compile test function complete!"

		if no_train:
			return
		# NLL cost
		cost = classifier.negative_log_likelihood(y) \
			 + self.l1_reg * classifier.L1 \
			 + self.l2_reg * classifier.L2_sqr

		gparams = []
		for param in classifier.params:
			gparam = T.grad(cost, param)
			gparams.append(gparam)

		updates = []
		#	C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
		for param, gparam in zip(classifier.params, gparams):
			updates.append((param, param - self.lr * gparam))
		# updates = {}
		# for param in classifier.params:
		# 	gparam = T.grad(cost, param)
		# 	updates[param] = param - self.lr * gparam

		# train model function
		self.train_batch = theano.function(inputs=[index], 
									# outputs=cost, 
									updates=updates,
									givens={
										x : self.train_data[index * self.batch_size : (index+1) * self.batch_size],
										y : self.label_data[index * self.batch_size : (index+1) * self.batch_size]
									})

		print "Compile training function complete!"	

	def __loadhvalues(self):
		'''
		@summary: Load the hidden output values of MlpNgram from given file
		'''
		backupfile = open(self.hvalue_file)
		self.hvalues = cPickle.load(backupfile)
		backupfile.close()

	def tids2inputdata(self, tidseq, zero_start=True, truncate_input = True):
		'''
		@summary: Specific data preprocess for MlpNgram. Concat N-1 vectors
		
		@param tidseq:
		@param zero_start: index start from the first token or not
		@param truncate_input: whether reserve the last token 
				(if True, the last token will be the last label, and will not use as last input) 
		@result: 
		'''
		in_size = len(tidseq)
		if not truncate_input:
			in_size += 1
		
		# zeros used to fill the empty input
		zeros = numpy.zeros((self.n_in,), dtype=theano.config.floatX)
	
		mat_in = []
		start_index = zero_start and 1 or (self.N - 1)
		for i in xrange(start_index, in_size):
			idxs = range(i - self.N + 1, i)
			vec = []
			for j in idxs:
				if j < 0:
					hvalue = zeros.copy()
				else:
					hvalue = self.hvalues[tidseq[j]]
				vec.extend(hvalue)
			mat_in.append(vec)
		mat_in = theano.shared(numpy.asarray(mat_in, dtype=theano.config.floatX), borrow=True)

		vec_out = zero_start and tidseq[1:] or tidseq[self.N-1:]
		vec_out = theano.shared(numpy.asarray(vec_out, dtype="int32"), borrow=True)

		return mat_in, vec_out

	def traintidseq(self, tidseq, data_slice_size=100000):
		'''
		@summary: Train a token id sequence. Slice data for GPU transpose data
		
		@param tidseq:
		@param data_slice_size:
		'''
		# init the share training data
		data_slice = tidseq[:data_slice_size+1]
		self.__setTrainData(self.tids2inputdata(data_slice))
		# init mlp
		self.__initMlp()
		# train the first time
		n_batch = int(math.ceil(data_slice_size / self.batch_size))
		for i in xrange(n_batch):
			self.train_batch(i)

		# train the rest data slices
		total_data_size = len(tidseq)
		for i in xrange(data_slice_size, total_data_size, data_slice_size):
			data_slice = tidseq[i-self.N+2:i+data_slice_size+1]
			self.__setTrainData(self.tids2inputdata(data_slice, zero_start=False))

			n_batch = int(math.ceil(data_slice_size / self.batch_size))
			for i in xrange(n_batch):
				self.train_batch(i)

	def traintext(self, text, test_text, add_se=False, epoch=100, DEBUG=False, SAVE=False, SINDEX=1):
		# token chars to token ids
		tidseq = self.tokens2ids(text, add_se)
		
		print "MlpNgram train start!!"
		s_time = time.clock()
		for i in xrange(epoch):
			self.traintidseq(tidseq)

			if DEBUG:
				print "Error rate: %0.5f. Epoch: %s. Training time so far: %0.1fm" % (self.testtext(test_text), i+SINDEX, (time.clock()-s_time)/60.)

			if SAVE:
				self.savemodel("./data/MlpNgram/Mlp%sgram.model.epoch%s.n_hidden%s.obj" % (self.N, i+SINDEX, self.n_hidden))

		e_time = time.clock()

		duration = e_time - s_time

		print "MlpNgram train over!! The total training time is %.2fm." % (duration / 60.) 	

	def testtext(self, text):

		self.__initMlp(no_train=True)
		# get input data
		mat_in, vec_out = self.tids2inputdata(self.tokens2ids(text))
		error = self.test_model(mat_in.get_value(borrow=True), vec_out.get_value(borrow=True))
		return error

	def predict(self, text):

		# text length should be large than 0
		if(len(text) < 1):
			return None

		self.__initMlp(no_train=True)

		# token to NN input and label
		tidmat, _ = self.tids2inputdata(self.tokens2ids(text[-1]), truncate_input=False)
		return self.mlp_predict(tidmat.get_value(borrow=True))

	def likelihood(self, text):

		# text length should be large than 1
		if(len(text) < 2):
			return None

		self.__initMlp(no_train=True)

		# token to NN input and label
		mat_in, vec_out = self.tids2inputdata(self.tokens2ids(text))
		return self.mlp_prob(mat_in.get_value(borrow=True), vec_out.get_value(borrow=True))


	def crossentropy(self, text, add_se=False):

		log_prob = numpy.log(self.likelihood(text))

		crossentropy = - numpy.mean(log_prob)

		return crossentropy

	def ranks(self, text):

		self.__initMlp(no_train=True)
		mat_in, label = self.tids2inputdata(self.tokens2ids(text))

		sort_matrix, probs = self.mlp_sort(mat_in.get_value(borrow=True), label.get_value(borrow=True))

		rank_list = []
		dict_size = self.ndict.size()
		label = label.get_value()
		for i in xrange(label.shape[0]):
			rank_list.append(dict_size - sort_matrix[i].searchsorted(probs[i]))

		return rank_list

	def logaverank(self, text):
		'''
		@summary: Return the average log rank
		
		@param text:
		@result: 
		'''
		log_ranks = numpy.log(self.ranks(text))

		return numpy.mean(log_ranks)

	def topN(self, text, N=10):
		'''
		@summary: Return the top N predict char of the history tids
		'''
		self.__initMlp(no_train=True)
		tidmat, _ = self.tids2inputdata(self.tokens2ids(text), truncate_input=False)
		probs = self.mlp_probs(tidmat.get_value(borrow=True))
		
		sort_probs = probs.copy()
		sort_probs.sort()
		top_probs = sort_probs[-N:][::-1]

		probs = list(probs)
		top_tokens = [probs.index(x) for x in top_probs]

		return top_tokens, top_probs

	def savemodel(self, filepath="./data/MlpNgram/MlpNgram.model.obj"):

		backupfile = open(filepath, 'w')
		cPickle.dump((self.batch_size, self.N, self.n_in, self.n_hidden, self.lr, self.l1_reg, self.l2_reg, self.mlpparams), backupfile)
		backupfile.close()
		print "Save model complete! Filepath:", filepath

	def loadmodel(self, filepath="./data/MlpNgram/MlpNgram.model.obj"):

		backupfile = open(filepath)
		self.batch_size, self.N, self.n_in, self.n_hidden, self.lr, self.l1_reg, self.l2_reg, self.mlpparams = cPickle.load(backupfile)
		backupfile.close()
		print "Load model complete! Filepath:", filepath



