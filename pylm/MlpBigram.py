# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 14:40
@summary: 
@author: egg
'''
import math
import numpy
import theano
import theano.tensor as T
import cPickle
from dltools.mlp import MLP
from theano import sandbox, Out

class MlpBigram(object):
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
		self.ndict = ndict

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
		x = T.matrix('x')  
		y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

		rng = numpy.random.RandomState(1234)

		# construct the MLP class
		self.mlp = MLP(rng=rng, input=x, n_in=self.ndict.size(),
						 n_hidden=self.n_hidden, n_out=self.ndict.size(),
						 hW=self.mlpparams[0],hb=self.mlpparams[1],oW=self.mlpparams[2],ob=self.mlpparams[3],)

		self.mlpparams = self.mlp.params

		classifier = self.mlp

		probs = classifier.logRegressionLayer.p_y_given_x[T.arange(y.shape[0]), y]
		self.mlp_prob = theano.function(inputs=[x, y], outputs=probs)
		print "Compile likelihood function complete!"

		y_pred = classifier.logRegressionLayer.y_pred
		self.mlp_predict = theano.function(inputs=[x], outputs=y_pred[-1])
		print "Compile predict function complete!"

		error = classifier.errors(y)
		# test model function
		self.test_model = theano.function(inputs=[x, y], outputs=error)
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
		
	def __tokens2ids(self, tokenseq, add_se=False):
		'''
		@summary: Token chars to token ids
		
		@param tokenseq: token chars
		@param add_se: whether add head and tail symbol
		@result: token ids
		'''
		tidseq = add_se and [self.ndict.getstartindex()] or []
		tidseq.extend([self.ndict[token] for token in tokenseq])
		add_se and tidseq.extend([self.ndict.getendindex()])

		# print tidseq
		return tidseq

	def __tids2nndata(self, tidseq, truncate_input=True):
		'''
		@summary: token ids to theano function input variables (matrix and vector)
		'''
		# print tidseq.shape
		in_size = len(tidseq)
		if truncate_input:
			in_size -= 1

		mat_in = numpy.zeros((in_size, self.ndict.size()), dtype=theano.config.floatX)
		vec_out = numpy.asarray(tidseq[1:], dtype="int32")

		for i in xrange(in_size):
			mat_in[i][tidseq[i]] = numpy.array(1., dtype=theano.config.floatX)

		mat_in = theano.shared(mat_in, borrow=True)
		vec_out = theano.shared(vec_out, borrow=True)

		return mat_in, vec_out

	def traintokenseq(self, tokenseq, add_se=False):
		'''
		@summary: Train a token sequence. Need to transform the tokens to tids
		
		@param tokenseq:
		'''

		# token to tids
		tidseq = self.__tokens2ids(tokenseq, add_se)

		# tids to theano input variables
		mat_in, vec_out = self.__tids2nndata(tidseq)

		self.train_batch(mat_in, vec_out)

	def traintext(self, text, add_se=False, data_slice_size=40000):
		'''
		@summary: Train text, split token sequence to slice with user-designed batch_size
		
		@param text: 
		'''
		
		# token chars to token ids
		tidseq = self.__tokens2ids(text, add_se)
		tidseq = theano.shared(tidseq).get_value(borrow=True)

		# train all slices of data. train_data and label_data will be reset to new slice
		total_data_size = len(tidseq)
		for i in xrange(0, total_data_size, data_slice_size):
			data_slice = tidseq[i:i+data_slice_size+1]
			self.train_data, self.label_data = self.__tids2nndata(data_slice)
			# print self.train_data, self.label_data
			# print self.train_data[1:2]
			self.__initMlp()

			n_batch = int(math.ceil(data_slice_size / self.batch_size))
			for i in xrange(n_batch):
				self.train_batch(i)

	def testtext(self, text):
		'''
		@summary: Test text, return the error rate of test text
		
		@param text:
		@result: Error rate
		'''

		self.__initMlp(no_train=True)

		# get input data
		mat_in, vec_out = self.__tids2nndata(self.__tokens2ids(text))

		error = self.test_model(mat_in.get_value(borrow=True), vec_out.get_value(borrow=True))

		return error

	def predict(self, text):
		'''
		@summary: Predict the next token, given the history text
		
		@param text:
		@result: 
		'''
		# text length should be large than 0
		if(len(text) < 1):
			return None

		self.__initMlp(no_train=True)

		# token to id
		tidmat, _ = self.__tids2nndata(self.__tokens2ids(text[-1]), truncate_input=False)
		return self.mlp_predict(tidmat.get_value(borrow=True))

	def likelihood(self, text):
		'''
		@summary: Return the probability of the last two token of the input text
		
		@param text:
		@result: 
		'''
		# text length should be large than 1
		if(len(text) < 2):
			return None

		self.__initMlp(no_train=True)

		# token to id
		mat_in, vec_out = self.__tids2nndata(self.__tokens2ids(text[-2:]))
		return self.mlp_prob(mat_in.get_value(borrow=True), vec_out.get_value(borrow=True))

	def crossentropy(self, text, add_se=False):
		'''
		@summary: Return the cross-entropy of the text
		
		@param text:
		@result: cross entropy
		'''

		# cal the log probability
		# cal cross-entropy first, use the equation:
		# 	H(W) = - 1 / N * (\sum P(w_1 w_2 ... w_N))
		log_prob = []
		len_seq = len(text)
		for i in xrange(len_seq-1):
			sub_seq = text[i:i+2]

			log_prob.append(numpy.log(self.likelihood(sub_seq)))

		crossentropy = - numpy.sum(log_prob) / len_seq

		return crossentropy, log_prob

	def savemodel(self, filepath="./data/MlpBigram.model.obj"):
		'''
		@summary: Save model to file
		
		@param filepath: back up file path
		'''

		backupfile = open(filepath, 'w')
		cPickle.dump((self.batch_size, self.n_hidden, self.lr, self.l1_reg, self.l2_reg, self.mlpparams), backupfile)
		backupfile.close()
		print "Save model complete!"

	def loadmodel(self, filepath="./data/MlpBigram.model.obj"):
		'''
		@summary: Save model to file
		
		@param filepath: back up file path
		'''

		backupfile = open(filepath)
		self.batch_size, self.n_hidden, self.lr, self.l1_reg, self.l2_reg, self.mlpparams = cPickle.load(backupfile)
		backupfile.close()
		print "Load model complete!"







