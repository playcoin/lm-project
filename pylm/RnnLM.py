# -*- coding: utf-8 -*-
'''
Created on 2013-05-05 21:46
@summary: RNN Language Model
@author: Playcoin
'''

import math
import numpy
import cPickle
import time

import theano
import theano.tensor as T
from theano import sandbox, Out

from LMBase import LMBase
from dltools.rnn import RNN

class RnnLM(LMBase):
	'''
	@summary: RNN Langeuage Model
	'''

	def __init__(self, ndict, n_hidden, lr, batch_size, backup_file_path=None, truncate_step=5):
		'''
		@summary: Construct function, initiate some attribute
		'''

		super(RnnLM, self).__init__(ndict)

		self.n_hidden = n_hidden
		self.lr = lr
		self.batch_size = batch_size
		self.truncate_step = truncate_step
		self.rnn_params = None

		self.rnn = None

	def __initRnn(self):
		'''
		@summary: Initiate RNN model 
		'''

		if self.rnn is not None:
			return

		x = T.tensor3('x')
		u = T.matrix('u')
		y = T.ivector('y')
		h_init = T.matrix('h_init')

		rng = numpy.random.RandomState(213234)
		rnn = RNN(rng, x, self.ndict.size(), self.n_hidden, self.ndict.size(), self.batch_size)
		self.rnn = rnn
		print "Compile Truncate-BPTT Algorithm!"
		rnn.build_tbptt(x, y, h_init, self.lr, self.truncate_step)

		print "Comile Test function"
		error = rnn.errors(u,y)
		self.test_model = theano.function([u, y], [error, rnn.y_pred])

	def traintext(self, text, add_se=True, epoch=200, DEBUG = False):

		self.__initRnn()

		test_text = text[20:200]

		# train whole text
		mat_in, label = self.tids2nndata(self.tokens2ids(text, add_se), shared=False)

		mat_in = mat_in.reshape(self.batch_size, mat_in.shape[0] / self.batch_size, mat_in.shape[1])
		mat_in = mat_in.transpose(1,0,2)
		# print seq_label
		label = label.reshape(self.batch_size, label.shape[0] / self.batch_size).T.flatten()

		mat_in, label = theano.shared(mat_in, borrow=True), theano.shared(label, borrow=True)

		s_time = time.clock()
		for i in xrange(epoch):
			self.rnn.train_tbptt(mat_in.get_value(), label.get_value())
			if DEBUG and (i + 1) % 10 == 0:
				err = self.testtext(test_text)[0]
				print "Error rate in epoch %s, is %.3f." % ( i+1, err)

		e_time = time.clock()
		print "RnnLM train over!! The total training time is %.2fm." % ((e_time - s_time) / 60.) 


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



