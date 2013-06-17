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
from dltools.mlp import MLP, MLPEMB
from theano import sandbox, Out
from LMBase import LMBase


class MlpNgram(LMBase):
	'''
	@summary: Mlp version for N-gram
	'''

	def __init__(self, ndict, N = 5, n_emb=50, n_hidden=200, lr=0.05, l1_reg = 0.00, l2_reg=0.0001, batch_size=40, dropout=False, emb_file_path=None, backup_file_path=None):
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
			self.n_in = ndict.size()
			self.n_emb = n_emb
			self.n_hidden = n_hidden
			self.lr = lr
			self.l1_reg = l1_reg
			self.l2_reg = l2_reg
			self.batch_size = batch_size
			self.mlpparams = [None, None, None, None, None]
		else:
			self.loadmodel(backup_file_path)

		# reload the embeddingfile
		if emb_file_path:
			f = open(emb_file_path)
			embvalues = cPickle.load(f)
			f.close()
			self.mlpparams[4] = theano.shared(embvalues, borrow=True)

		# mlp obj
		self.mlp = None
		self.n_emb = n_emb
		self.dropout = dropout

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
		x = T.imatrix('x')  
		y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

		rng = numpy.random.RandomState(4321)

		# construct the MLP class
		self.mlp = MLPEMB(rng = rng, input=x, n_in=self.n_in, n_token=self.N - 1,
						 n_emb = self.n_emb, n_hidden=self.n_hidden, n_out=self.n_in,
						 dropout = self.dropout,
						 hW=self.mlpparams[0], hb=self.mlpparams[1], 
						 oW=self.mlpparams[2], ob=self.mlpparams[3], 
						 C=self.mlpparams[4])

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

		self.lr = theano.shared(numpy.array(self.lr, dtype=theano.config.floatX), borrow=True)
		updates = []
		#	C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
		for param, gparam in zip(classifier.params, gparams):
			updates.append((param, param - self.lr * gparam))

		# train model function
		self.train_batch = theano.function(inputs=[index], 
									# outputs=cost, 
									updates=updates,
									givens={
										x : self.train_data[index * self.batch_size : (index+1) * self.batch_size],
										y : self.label_data[index * self.batch_size : (index+1) * self.batch_size]
									})

		print "Compile training function complete!"	

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
		mat_in = []
		start_index = zero_start and 1 or (self.N - 1)
		for i in xrange(start_index, in_size):
			idxs = range(i - self.N + 1, i)
			vec = []
			for j in idxs:
				if j < 0:
					hvalue = 0
				else:
					hvalue = tidseq[j]
				vec.append(hvalue)
			mat_in.append(vec)

		mat_in = theano.shared(numpy.asarray(mat_in, dtype="int32"), borrow=True)

		vec_out = zero_start and tidseq[1:] or tidseq[self.N-1:]
		vec_out = theano.shared(numpy.asarray(vec_out, dtype="int32"), borrow=True)

		return mat_in, vec_out

	def traintext(self, text, test_text, add_se=False, epoch=100, lr_coef = -1., DEBUG=False, SAVE=False, SINDEX=1):
		# token chars to token ids
		tidseq = self.tokens2ids(text, add_se)
		train_size = len(tidseq) - 1
		n_batch = int(math.ceil(train_size / self.batch_size))
		
		self.__setTrainData(self.tids2inputdata(tidseq))
		self.__initMlp()
		test_in, test_out = self.tids2inputdata(self.tokens2ids(test_text))
		print "MlpNgram model init complete!!"

		s_time = time.clock()
		for i in xrange(epoch):

			for idx in xrange(n_batch):
				self.train_batch(idx)

			if DEBUG:
				error = self.test_model(test_in.get_value(borrow=True), test_out.get_value(borrow=True))
				print "Error rate: %0.5f. Epoch: %s. Training time so far: %0.1fm" % (error, i+SINDEX, (time.clock()-s_time)/60.)

			if SAVE:
				self.savemodel("./data/MlpNgram/Mlp%sgram.model.epoch%s.n_hidden%s.dr%s.n_emb%s.in_size%s.obj" % (self.N, i+SINDEX, self.n_hidden, self.dropout, self.n_emb, self.n_in))
			
			if lr_coef > 0:
				# update learning_rate
				lr = self.lr.get_value(borrow=True) * lr_coef
				self.lr.set_value(numpy.array(lr, dtype=theano.config.floatX))

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

	def likelihood(self, sentence, debug=False):

		self.__initMlp(no_train=True)

		# token to NN input and label
		sentence = '\n' + sentence.strip() + '\n'
		mat_in, vec_out = self.tids2inputdata(self.tokens2ids(sentence))

		probs = self.mlp_prob(mat_in.get_value(borrow=True), vec_out.get_value(borrow=True))

		if debug:
			for i in range(len(probs)):
				print "	%s: %.5f" % (sentence[i+1] == '\n' and '<s>' or sentence[i+1], probs[i])

		return probs

	def ranks(self, sentence):

		self.__initMlp(no_train=True)
		sentence = '\n' + sentence.strip() + '\n'
		mat_in, label = self.tids2inputdata(self.tokens2ids(sentence))

		sort_matrix, probs = self.mlp_sort(mat_in.get_value(borrow=True), label.get_value(borrow=True))

		rank_list = []
		dict_size = self.ndict.size()
		label = label.get_value()
		for i in xrange(label.shape[0]):
			rank_list.append(dict_size - sort_matrix[i].searchsorted(probs[i]))

		return rank_list

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
		cPickle.dump((self.batch_size, self.N, self.n_in, self.n_hidden, self.lr.get_value(), self.l1_reg, self.l2_reg, self.mlpparams), backupfile)
		backupfile.close()
		print "Save model complete! Filepath:", filepath

	def loadmodel(self, filepath="./data/MlpNgram/MlpNgram.model.obj"):

		backupfile = open(filepath)
		self.batch_size, self.N, self.n_in, self.n_hidden, self.lr, self.l1_reg, self.l2_reg, self.mlpparams = cPickle.load(backupfile)
		backupfile.close()
		print "Load model complete! Filepath:", filepath



