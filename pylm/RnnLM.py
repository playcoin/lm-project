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

	def __init__(self, ndict, n_hidden, lr, batch_size, dropout=False, backup_file_path=None, truncate_step=5):
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
		self.dropout = dropout
		self.in_size = self.out_size  = ndict.size()

	def initRnn(self, no_train=False):
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
		rnn = RNN(rng, self.in_size, self.n_hidden, self.out_size, self.batch_size, self.lr, dropout=self.dropout, params=self.rnnparams)
		self.rnn = rnn
		self.rnnparams = rnn.params

		error = rnn.errors(u,y)
		self.test_model = theano.function([u, y], [error, rnn.y_pred])
		print "Compile Test function complete!"

		probs = rnn.y_prob[T.arange(y.shape[0]), y]
		self.rnn_prob = theano.function([u, y], probs)
		print "Compile likelihood function complete!"

		self.rnn_sort = theano.function([u, y], [rnn.y_sort_matrix, probs])
		print "Compile argsort function complete!"

		self.rnn_pred = theano.function([u], [rnn.y_pred[-1], rnn.y_prob[-1]])
		print "Compile predict function complete!"

		if not no_train:
			rnn.build_tbptt(x, y, h_init, self.truncate_step)
			print "Compile Truncate-BPTT Algorithm complete!"

	def reshape(self, in_data, l_data, s_idx, e_idx, len_sent):
		s_in = in_data[s_idx:e_idx].reshape(self.batch_size, len_sent, in_data.shape[1]).transpose(1, 0, 2)
		s_l = l_data[s_idx:e_idx].reshape(self.batch_size, len_sent).T.flatten()
		return s_in, s_l

	def traintext(self, text, test_text, add_se=True, sen_slice_length=4, epoch=200, lr_coef = -1., DEBUG = False, SAVE=False, SINDEX=1):

		self.initRnn()

		tidseq = self.tokens2ids(text, add_se)
		seq_size = len(tidseq)

		# variables for slice sentence
		sentence_length = sen_slice_length * self.truncate_step
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
				mat_in, label = self.reshape(mat_in_total, label_total, j, j+data_slice_size, sentence_length)
				self.rnn.train_tbptt(mat_in, label)

				if (j+data_slice_size+half_sen_length) < seq_size:
					mat_in, label = self.reshape(mat_in_total, label_total, j+half_sen_length, j+data_slice_size+half_sen_length, sentence_length)
					self.rnn.train_tbptt(mat_in, label)
			
			if DEBUG:
				err = self.testtext(test_text)[0]
				e_time = time.clock()
				print "Error rate in epoch %s, is %.3f. Training time so far is: %.2fm" % ( i+SINDEX, err, (e_time-s_time) / 60.)
				# print ''.join([self.ndict.gettoken(x) for x in r_labels])

			if SAVE:
				self.savemodel("./data/RnnLM/RnnLM.model.epoch%s.n_hidden%s.truncstep%s.obj" % (i+SINDEX, self.n_hidden, self.truncate_step))

			if lr_coef > 0:
				# update learning_rate
				lr = self.rnn.lr.get_value() * lr_coef
				self.rnn.lr.set_value(numpy.array(lr, dtype=theano.config.floatX))

		e_time = time.clock()
		print "RnnLM train over!! The total training time is %.2fm." % ((e_time - s_time) / 60.) 

	def testtext(self, text):

		self.initRnn(no_train=True)

		mat_in, label = self.tids2nndata(self.tokens2ids(text), shared=False)

		return self.test_model(mat_in, label)

	def predict(self, text):
		self.initRnn(no_train=True)
		mat_in, _ = self.tids2nndata(self.tokens2ids(text), truncate_input=False, shared=False)

		return self.rnn_pred(mat_in)

	def likelihood(self, text):

		self.initRnn(no_train=True)
		mat_in, label = self.tids2nndata(self.tokens2ids(text), shared=False)

		return self.rnn_prob(mat_in, label)


	def crossentropy(self, text):

		log_probs = numpy.log(self.likelihood(text))

		crossentropy = - numpy.mean(log_probs)

		return crossentropy

	def ranks(self, text):
		self.initRnn(no_train=True)
		mat_in, label = self.tids2nndata(self.tokens2ids(text), shared=False)

		sort_matrix, probs = self.rnn_sort(mat_in, label)

		rank_list = []
		dict_size = self.ndict.size()
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
		_, probs = self.predict(text)
		
		sort_probs = probs.copy()
		sort_probs.sort()
		top_probs = sort_probs[-N:][::-1]

		probs = list(probs)
		top_tokens = [probs.index(x) for x in top_probs]

		return top_tokens, top_probs


	def savemodel(self, filepath="./data/RnnLM.model.obj"):
		backupfile = open(filepath, 'w')
		cPickle.dump((self.batch_size, self.n_hidden, self.lr, self.truncate_step, self.rnnparams), backupfile)
		backupfile.close()
		print "Save model complete! Filepath:", filepath

	def loadmodel(self, filepath="./data/RnnLM.model.obj"):
		backupfile = open(filepath)
		self.batch_size, self.n_hidden, self.lr, self.truncate_step, self.rnnparams = cPickle.load(backupfile)
		backupfile.close()
		print "Load model complete! Filepath:", filepath


