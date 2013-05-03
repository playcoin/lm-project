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
from dltools.mlp import MLP
from theano import sandbox, Out

class MlpBigram(object):
	'''
	@summary: Train Bigram by using Mlp
	'''

	def __init__(self, ndict, n_hidden=30, lr=0.05, l1_reg = 0.00, l2_reg=0.0001, batch_size=40):
		'''
		@summary: Construct function, set some parameters.
		
		@param ndict: NDict obj
		@param lr: learing rate
		@param l1_reg: l1 norm coef
		@param l2_reg: l2 norm coef
		'''

		self.ndict = ndict
		self.n_hidden = n_hidden
		self.lr = lr
		self.l1_reg = l1_reg
		self.l2_reg = l2_reg
		self.batch_size = batch_size

		# mlp obj
		self.mlp = None


	def __initMlp(self):
		'''
		@summary: Init mlp, build training and test function
		@result: 
		'''
		if self.mlp:
			return

		print "Init theano symbol expressions!"

		index = T.lscalar()
		x = T.matrix('x')  
		y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

		rng = numpy.random.RandomState(1234)

		# construct the MLP class
		self.mlp = MLP(rng=rng, input=x, n_in=self.ndict.size(),
						 n_hidden=self.n_hidden, n_out=self.ndict.size())

		classifier = self.mlp

		error = classifier.errors(y)
		# test model function
		self.test_model = theano.function(inputs=[x, y], outputs=[error, classifier.logRegressionLayer.y_pred])
		print "Compile test function complete!"

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
									outputs=Out(cost, borrow=True), 
									updates=updates,
									givens={
										x : self.train_data[index * self.batch_size : (index+1) * self.batch_size],
										y : self.label_data[index * self.batch_size : (index+1) * self.batch_size]
									},
									mode='ProfileMode')
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

	def __tids2nndata(self, tidseq):
		'''
		@summary: token ids to theano function input variables (matrix and vector)
		'''
		# print tidseq.shape
		in_size = len(tidseq) - 1

		mat_in = numpy.zeros((in_size, self.ndict.size()), dtype=theano.config.floatX)
		mat_out = numpy.asarray(tidseq[1:], dtype="int32")

		for i in xrange(in_size):
			mat_in[i][tidseq[i]] = numpy.array(1., dtype=theano.config.floatX)

		mat_in = theano.shared(mat_in)
		mat_out = theano.shared(mat_out)

		return mat_in, mat_out

	def traintokenseq(self, tokenseq, add_se=False):
		'''
		@summary: Train a token sequence. Need to transform the tokens to tids
		
		@param tokenseq:
		'''

		# token to tids
		tidseq = self.__tokens2ids(tokenseq, add_se)

		# tids to theano input variables
		mat_in, mat_out = self.__tids2nndata(tidseq)

		self.train_batch(mat_in, mat_out)

	def traintext(self, text, add_se=False, data_slice_size=20000):
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

		self.__initMlp()

		# get input data
		mat_in, mat_out = self.__tids2nndata(self.__tokens2ids(text))

		# print mat_in, mat_out

		error = self.test_model(mat_in.get_value(borrow=True), mat_out.get_value(borrow=True))

		return error








