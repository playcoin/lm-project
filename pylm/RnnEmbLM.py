# -*- coding: utf-8 -*-
'''
Created on 2013-06-05 15:27
@summary: Rnn language Model use Character Embedding
@author: egg
'''
import math
import numpy
import cPickle
import time

import theano
import theano.tensor as T

from RnnLM import RnnLM
from LMBase import LMBase
from dltools.rnn import RNN
from dltools.rnnemb import RNNEMB

class RnnEmbLM(RnnLM):
	'''
	@summary: Rnn language Model use Character Embedding
	'''
	def __init__(self, *args, **kwargs):
		'''
		@summary: construct method
		
		@param *args:
		@param **kwargs:
		'''
		super(RnnEmbLM, self).__init__(*args, **kwargs)

		self.in_size = 50

	def tids2nndata(self, tidseq, truncate_input = True, shared =False):
		# print tidseq.shape
		seq_size = len(tidseq)
		if truncate_input:
			seq_size -= 1

		mat_in = []
		for i in xrange(seq_size):
			mat_in.append(numpy.array(self.embvalues[tidseq[i]], dtype=theano.config.floatX))

		mat_in = numpy.asarray(mat_in)
		vec_out = numpy.asarray(tidseq[1:], dtype="int32")

		return mat_in, vec_out

	def loadEmbeddings(self, filepath):
		'''
		@summary: load embvalues
		
		@param filepath:
		@result: 
		'''
		backupfile = open(filepath)
		self.embvalues = cPickle.load(backupfile)
		backupfile.close()

class RnnEmbTrLM(RnnLM):
	'''
	@summary: Rnn language Model use Character Embedding, and ajust embedding in the training time.
	'''

	def __init__(self, ndict, n_hidden, lr, batch_size, train_emb=True, emb_file_path = None, dropout=False, truncate_step=5, backup_file_path=None):

		LMBase.__init__(self, ndict)

		if backup_file_path is None:
			self.n_hidden = n_hidden
			self.lr = lr
			self.batch_size = batch_size
			self.truncate_step = truncate_step
			self.rnnparams = None
			self.embvalues = None
		else:
			self.loadmodel(backup_file_path)

		self.rnn = None
		self.dropout = dropout
		self.train_emb = train_emb
		self.in_size = self.out_size  = ndict.size()

		if emb_file_path:
			f = open(emb_file_path)
			self.embvalues = cPickle.load(f)
			f.close()
			self.embvalues = self.embvalues.astype(theano.config.floatX)
			
		self.embsize = (self.embvalues is None) and self.in_size or self.embvalues.shape[1]

	def initRnn(self, no_train=False):
		'''
		@summary: Initiate RNNEMB model 
		'''
		if self.rnn is not None:
			return

		print "RnnEmbTrLM!"
		x = T.imatrix('x')
		u = T.ivector('u')
		y = T.ivector('y')
		h_init = T.matrix('h_init')

		rng = numpy.random.RandomState(213234)
		rnn = RNNEMB(rng, 
				self.in_size,
				self.embsize, 
				self.n_hidden, 
				self.ndict.size(),
				self.batch_size,
				self.lr,
				self.dropout,
				params = self.rnnparams,
				embeddings = self.embvalues
			)

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
			rnn.build_tbptt(x, y, h_init, self.truncate_step, self.train_emb)
			print "Compile Truncate-BPTT Algorithm complete!"

	def tids2nndata(self, tidseq, truncate_input = True, shared =False):
		# print tidseq.shape
		seq_size = len(tidseq)
		if truncate_input:
			seq_size -= 1

		vec_in = numpy.asarray(tidseq[:seq_size], dtype="int32")
		vec_out = numpy.asarray(tidseq[1:], dtype="int32")

		return vec_in, vec_out

	def reshape(self, in_data, l_data, s_idx, e_idx, len_sent):
		s_in = in_data[s_idx:e_idx].reshape(self.batch_size, len_sent).T
		s_l = l_data[s_idx:e_idx].reshape(self.batch_size, len_sent).T.flatten()
		return s_in, s_l

	def savemodel(self, filepath):
		backupfile = open(filepath, 'w')
		cPickle.dump((self.batch_size, self.n_hidden, self.lr, self.truncate_step, self.rnnparams, self.rnn.C.get_value()), backupfile)
		backupfile.close()
		print "Save model complete! Filepath:", filepath

	def loadmodel(self, filepath):
		backupfile = open(filepath)
		self.batch_size, self.n_hidden, self.lr, self.truncate_step, self.rnnparams, self.embvalues = cPickle.load(backupfile)
		backupfile.close()
		print "Load model complete! Filepath:", filepath
