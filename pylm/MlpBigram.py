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

	def __init__(self, ndict, n_hidden=30, lr=0.13, l1_reg = 0.01, l2_reg=0.):
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

		x = T.matrix('x')  # the data is presented as rasterized images
		y = T.ivector('y')  # the labels are presented as 1D vector of
							# [int] labels

		rng = numpy.random.RandomState(1234)

		# construct the MLP class
		self.mlp = MLP(rng=rng, input=x, n_in=self.ndict.size(),
						 n_hidden=self.n_hidden, n_out=self.ndict.size())

		classifier = self.mlp

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

		train_model = theano.function(inputs=[x, y], outputs=cost, updates=updates)
		print "Compile training function complete!"

		self.train_batch = train_model

	def traintokenseq(self, tokenseq, add_se=True):
		'''
		@summary: Train a token sequence. Need to transform the tokens to tids
		
		@param tokenseq:
		'''

		self.__initMlp()

		# token to tids
		tidseq = add_se and [self.ndict.getstartindex()] or []
		tidseq.extend([self.ndict[token] for token in tokenseq])
		add_se and tidseq.extend([self.ndict.getendindex()])

		# tids to matrix
		mat_in = numpy.zeros((len(tidseq)-1, self.ndict.size()), dtype=theano.config.floatX)
		mat_out = numpy.asarray(tidseq[1:], dtype=theano.config.floatX)

		for i in xrange(len(tidseq)-1):
			mat_in[i][tidseq[i]] = 1.

		print self.train_batch(mat_in, mat_out)








