# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 14:40
@summary: 
@author: egg
'''
import math
import time
import numpy
import theano
import theano.tensor as T
import cPickle
from dltools.mlp import MLP
from theano import sandbox, Out
from LMBase import LMBase

class MlpBigram(LMBase):
	'''
	@summary: Train Bigram by using Mlp
	'''

	def __init__(self, ndict, n_hidden=30, lr=0.05, l1_reg = 0.00, l2_reg=0.0001, batch_size=40, backup_file_path=None):
		'''
		@summary: Construct function, set some parameters.
		
		@param ndict: NDict obj
		@param lr: learing rate
		@param l1_reg: l1 norm coef
		@param l2_reg: l2 norm coef
		'''
		super(MlpBigram, self).__init__(ndict)

		if backup_file_path is None:
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


	def __initMlp(self, no_train=False):
		'''
		@summary: Init mlp, build training and test function
		@result: 
		'''
		if self.mlp is not None:
			return

		print "Init theano symbol expressions!"

		index = T.lscalar()
		x = T.ivector('x')  
		y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

		rng = numpy.random.RandomState(1234)

		# construct the MLP class
		self.mlp = MLP(rng=rng, input=x, n_in=self.ndict.size(),
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

		self.mlp_hidden = theano.function(inputs=[x], outputs=classifier.hiddenLayer.output)
		print "Compile hidden output function complete!"

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

	def tids2nndata(self, tidseq, truncate_input = True, shared=True):
		# print tidseq.shape
		seq_size = len(tidseq)
		if truncate_input:
			seq_size -= 1

		# for CUDA
		vec_in = theano.shared(numpy.asarray(tidseq[:seq_size], dtype="int32"), borrow=True)
		vec_out = theano.shared(numpy.asarray(tidseq[1:], dtype="int32"), borrow=True)

		if shared:
			return vec_in, vec_out
		else:
			return vec_in.get_value(borrow=True), vec_out.get_value(borrow=True)
		
	def traintext(self, text, test_text, add_se=False, epoch=100, DEBUG=False, SAVE=False, SINDEX=1):
		
		# token chars to token ids
		tidseq = self.tokens2ids(text, add_se)
		self.train_data, self.label_data = self.tids2nndata(tidseq, shared=True)
		self.__initMlp()
		data_size = len(tidseq)
		n_batch = int(math.ceil(data_size / self.batch_size))
		
		print "MlpBigram train start!!"
		s_time = time.clock()
		for i in xrange(epoch):
			
			for j in xrange(n_batch):
				self.train_batch(j)

			if DEBUG:
				print "Error rate: %0.5f. Epoch: %s. Training time so far: %0.1fm" % (self.testtext(test_text), i+SINDEX, (time.clock()-s_time)/60.)

			if SAVE:
				self.savemodel("./data/MlpBigram/MlpBigram.model.epoch%s.n_hidden%s.obj" % (i+SINDEX, self.n_hidden))

		e_time = time.clock()

		duration = e_time - s_time

		print "MlpBigram train over!! The total training time is %.2fm." % (duration / 60.) 	

	def testtext(self, text):

		self.__initMlp(no_train=True)

		# get input data
		mat_in, vec_out = self.tids2nndata(self.tokens2ids(text))

		error = self.test_model(mat_in.get_value(borrow=True), vec_out.get_value(borrow=True))

		return error

	def predict(self, text):

		# text length should be large than 0
		if(len(text) < 1):
			return None

		self.__initMlp(no_train=True)

		# token to NN input and label
		tidmat, _ = self.tids2nndata(self.tokens2ids(text[-1]), truncate_input=False)
		return self.mlp_predict(tidmat.get_value(borrow=True))

	def likelihood(self, sentence):

		self.__initMlp(no_train=True)
		sentence = '\n' + sentence.strip() + '\n'

		# token to NN input and label
		mat_in, vec_out = self.tids2nndata(self.tokens2ids(sentence), shared=False)
		return self.mlp_prob(mat_in, vec_out)

	def ranks(self, sentence):

		self.__initMlp(no_train=True)
		sentence = '\n' + sentence.strip() + '\n'
		mat_in, label = self.tids2nndata(self.tokens2ids(sentence), shared=False)

		sort_matrix, probs = self.mlp_sort(mat_in, label)

		rank_list = []
		dict_size = self.ndict.size()
		for i in xrange(label.shape[0]):
			rank_list.append(dict_size - sort_matrix[i].searchsorted(probs[i]))

		return rank_list

	def topN(self, text, N=10):
		'''
		@summary: Return the top N predict char of the history tids
		'''
		self.__initMlp(no_train=True)
		tidmat, _ = self.tids2nndata(self.tokens2ids(text), truncate_input=False)
		probs = self.mlp_probs(tidmat.get_value(borrow=True))
		
		sort_probs = probs.copy()
		sort_probs.sort()
		top_probs = sort_probs[-N:][::-1]

		probs = list(probs)
		top_tokens = [probs.index(x) for x in top_probs]

		return top_tokens, top_probs

	def savehvalues(self, filepath="./data/MlpBigram.hiddens.obj"):
		'''
		@summary: Output the output value of hidden layer by cPickle
		
		@param filepath:
		'''

		self.__initMlp(no_train = True)

		tids = range(0, self.ndict.size())

		mat_in, _ = self.tids2nndata(tids, truncate_input=False)

		hidden_values = self.mlp_hidden(mat_in.get_value())

		backupfile = open(filepath, 'w')
		cPickle.dump(hidden_values, backupfile)
		backupfile.close()
		print "Save hidden values complete!"

	def savemodel(self, filepath="./data/MlpBigram.model.obj"):

		backupfile = open(filepath, 'w')
		cPickle.dump((self.batch_size, self.n_hidden, self.lr, self.l1_reg, self.l2_reg, self.mlpparams), backupfile)
		backupfile.close()
		print "Save model complete! Filepath:", filepath

	def loadmodel(self, filepath="./data/MlpBigram.model.obj"):

		backupfile = open(filepath)
		self.batch_size, self.n_hidden, self.lr, self.l1_reg, self.l2_reg, self.mlpparams = cPickle.load(backupfile)
		backupfile.close()
		print "Load model complete! Filepath:", filepath







