# -*- coding: utf-8 -*-
'''
Created on 2013-05-05 21:46
@summary: RNN Language Model
@author: Playcoin
'''

import math
import numpy
import cPickle

import theano
import theano.tensor as T
from theano import sandbox, Out

from LMBase import LMBase
from dltools.rnn import RNN

class RnnLM(LMBase):
	'''
	@summary: RNN Langeuage Model
	'''

	def __init__(self, ndict, n_hidden, lr, batch_size, backup_file_path=None):
		'''
		@summary: Construct function, initiate some attribute
		'''

		super(RnnLM, self).__init__(ndict)

		self.n_hidden = n_hidden
		self.lr = lr
		self.batch_size = batch_size
		self.rnn_params = None

		self.rnn = None

	def __initRnn(self):
		'''
		@summary: Initiate RNN model 
		'''

		if self.rnn is not None:
			return

		x = T.matrix('x')
		y = T.ivector('y')
		h_init = T.vector('h_init')

		rng = numpy.random.RandomState(213234)
		rnn = RNN(rng, x, self.ndict.size(), self.n_hidden, self.ndict.size())
		self.rnn = rnn
		print "Compile Truncate-BPTT Algorithm!"
		rnn.build_tbptt(x, y, h_init, self.lr)

		print "Comile test function"
		error = rnn.errors(y)
		self.test_model = theano.function([x, y], [error, rnn.y_pred])

	def traintext(self, text, add_se=True):

		self.__initRnn()
		# # train each sentence separately
		# sentences = text.split('\n')

		# for sentence in sentences:
		# 	# print sentence
		# 	mat_in, label = self.tids2nndata(self.tokens2ids(sentence, add_se))
		# 	self.rnn.train_tbptt(mat_in.get_value(), label.get_value(), self.lr)

		# train whole text
		mat_in, label = self.tids2nndata(self.tokens2ids(text, add_se))
		self.rnn.train_tbptt(mat_in.get_value(), label.get_value(), self.lr)

	def testtext(self, text):

		mat_in, label = self.tids2nndata(self.tokens2ids(text))

		return self.test_model(mat_in.get_value(), label.get_value())

	def predict(self, text):
		return

	def likelihood(self, text):
		return

	def crossentropy(self, text):
		return

	def savemodel(self, filepath="./data/MlpBigram.model.obj"):
		# TODO
		pass

	def loadmodel(self, filepath="./data/MlpBigram.model.obj"):
		pass



