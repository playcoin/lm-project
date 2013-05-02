# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 14:40
@summary: 
@author: egg
'''
import numpy
import theano
import theano.tensor as T
from dltools.mlp import MLP

class MlpBigram(object):
	'''
	@summary: Train Bigram by using Mlp
	'''

	def __init__(self, ndict, n_hidden=30, lr=0.05, l1_reg = 0.00, l2_reg=0.0001):
		'''
		@summary: Construct function, set some parameters.
		
		@param ndict: NDict obj
		@param lr: learing rate
		@param l1_reg: l1 norm coef
		@param l2_reg: l2 norm coef
		'''

		self.ndict = ndict;
		self.n_hidden = n_hidden;
		self.lr = lr;
		self.l1_reg = l1_reg;
		self.l2_reg = l2_reg;

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

		x = T.matrix('x')  
		y = T.vector('y', dtype="int64")  # the labels are presented as 1D vector of [int] labels

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
		self.train_batch = theano.function(inputs=[x, y], outputs=[cost, classifier.logRegressionLayer.y_pred], updates=updates)
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
		mat_in = numpy.zeros((len(tidseq)-1, self.ndict.size()), dtype=theano.config.floatX)
		mat_out = numpy.asarray(tidseq[1:], dtype="int64")

		for i in xrange(len(tidseq)-1):
			mat_in[i][tidseq[i]] = numpy.asarray([1.], dtype=theano.config.floatX).item(0)

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

	def traintext(self, text, batch_size=50, add_se=False):
		'''
		@summary: Train text, split token sequence to slice with user-designed batch_size
		
		@param text: 
		'''
		self.__initMlp()

		train_size = len(text)
		for i in xrange(0, train_size, batch_size):
			train_slice = text[i:i+batch_size+1]
			self.traintokenseq(train_slice, add_se)

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

		error = self.test_model(mat_in, mat_out)

		return error








