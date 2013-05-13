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

		if backup_file_path is None:
			self.n_hidden = n_hidden
			self.lr = lr
			self.batch_size = batch_size
			self.truncate_step = truncate_step
			self.rnnparams = None
		else:
			self.loadmodel(backup_file_path)

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
		rnn = RNN(rng, x, self.ndict.size(), self.n_hidden, self.ndict.size(), self.batch_size, params=self.rnnparams)
		self.rnn = rnn
		print "Compile Truncate-BPTT Algorithm!"
		rnn.build_tbptt(x, y, h_init, self.lr, self.truncate_step)

		print "Comile Test function"
		error = rnn.errors(u,y)
		self.test_model = theano.function([u, y], [error, rnn.y_pred])

	def traintext(self, text, test_text, add_se=True, epoch=200, DEBUG = False, SAVE=False):

		self.__initRnn()

		tidseq = self.tokens2ids(text, add_se)
		seq_size = len(tidseq)

		# variables for slice sentence
		sentence_length = 4 * self.truncate_step
		half_sen_length = sentence_length / 2
		data_slice_size = sentence_length * self.batch_size
		data_size = seq_size / data_slice_size * data_slice_size
		print "Data size:" , data_size
		data_size -= data_slice_size / 2

		mat_in_total, label_total = self.tids2nndata(tidseq, shared=False)

		s_time = time.clock()
		for i in xrange(epoch):

			for j in xrange(0, data_size, data_slice_size / 2):
				# reshape to matrix: [SEQ][BATCH][FRAME]
				# mat_in, label = self.tids2nndata(tidseq[j:j+data_slice_size+1], shared=False)
				mat_in = mat_in_total[j:j+data_slice_size].reshape(self.batch_size, sentence_length, mat_in_total.shape[1]).transpose(1,0,2)
				label = label_total[j:j+data_slice_size].reshape(self.batch_size, sentence_length).T.flatten()
				# mat_in, label = theano.shared(mat_in, borrow=True), theano.shared(label, borrow=True)
				
				self.rnn.train_tbptt(mat_in, label)

				if (j+data_slice_size+half_sen_length) < seq_size:
					mat_in = mat_in_total[j+half_sen_length:j+data_slice_size+half_sen_length].reshape(self.batch_size, sentence_length, mat_in_total.shape[1]).transpose(1,0,2)
					label = label_total[j+half_sen_length:j+data_slice_size+half_sen_length].reshape(self.batch_size, sentence_length).T.flatten()
					# mat_in, label = theano.shared(mat_in, borrow=True), theano.shared(label, borrow=True)
					
					self.rnn.train_tbptt(mat_in, label)
			
			if DEBUG:
				err, r_labels = self.testtext(test_text)
				e_time = time.clock()
				print "Error rate in epoch %s, is %.3f. Training time so far is: %.2fm" % ( i+1, err, (e_time-s_time) / 60.)
				print ''.join([self.ndict.gettoken(x) for x in r_labels])

			if SAVE:
				self.savemodel("./data/RnnLM.model.epoch%s.obj" % i+1)

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

	def savemodel(self, filepath="./data/RnnLM.model.obj"):
		backupfile = open(filepath, 'w')
		cPickle.dump((self.batch_size, self.n_hidden, self.lr, self.truncate_step, self.mlpparams), backupfile)
		backupfile.close()
		print "Save model complete!"

	def loadmodel(self, filepath="./data/RnnLM.model.obj"):
		backupfile = open(filepath)
		self.batch_size, self.n_hidden, self.lr, self.truncate_step, self.mlpparams = cPickle.load(backupfile)
		backupfile.close()
		print "Load model complete!"


