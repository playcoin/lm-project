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
		l = T.imatrix('l')
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
		self.test_model = theano.function([u, y], [error])
		print "Compile Test function complete!"

		probs = rnn.y_prob[T.arange(y.shape[0]), y]
		self.rnn_prob = theano.function([u, y], probs)
		print "Compile likelihood function complete!"

		self.rnn_sort = theano.function([u, y], [rnn.y_sort_matrix, probs])
		print "Compile argsort function complete!"

		self.rnn_pred = theano.function([u], [rnn.y_pred[-1], rnn.y_prob[-1]])
		print "Compile predict function complete!"

	def tids2nndata(self, tidseq, truncate_input = True, shared=False):
		# print tidseq.shape
		seq_size = len(tidseq)
		if truncate_input:
			seq_size -= 1

		vec_in = numpy.asarray(tidseq[:seq_size], dtype="int32")
		vec_out = numpy.asarray(tidseq[1:], dtype="int32")

		if shared:
			return theano.shared(vec_in, borrow=True).get_value(), theano.shared(vec_out, borrow=True).get_value()

		return vec_in, vec_out

	def reshape(self, in_data, l_data):
		s_in = in_data.reshape(self.batch_size, in_data.shape[0] / self.batch_size).T
		s_l = l_data.reshape(self.batch_size,  l_data.shape[0] / self.batch_size).T
		return theano.shared(s_in, borrow=True), theano.shared(s_l, borrow=True)

	def traintext(self, text, test_text, add_se=True, sen_slice_length=4, epoch=200, lr_coef = -1., DEBUG = False, SAVE=False, SINDEX=1):

		self.initRnn()

		tidseq = self.tokens2ids(text, add_se)
		seq_size = len(tidseq)

		# variables for slice sentence
		sentence_length = sen_slice_length * self.truncate_step
		half_sen_length = sentence_length / 2
		data_slice_size = sentence_length * self.batch_size
		data_size = seq_size / data_slice_size * data_slice_size

		mat_in_total, label_total = self.tids2nndata(tidseq, shared=True)
		re_mat_in, re_label = self.reshape(mat_in_total[:data_size], label_total[:data_size])
		self.rnn.build_tbptt(re_mat_in, re_label, self.truncate_step, self.train_emb)
		print "Compile Truncate-BPTT Algorithm complete!"

		ttids = self.tokens2ids(test_text, add_se)
		t_in, t_l = self.tids2nndata(ttids, shared=True)

		# total sentence length after reshape
		total_sent_len = data_size / self.batch_size
		print "Actural training data size: %s, with total sentence length: %s" % (data_size, total_sent_len)
		s_time = time.clock()
		for i in xrange(epoch):

			for j in xrange(0, total_sent_len - half_sen_length, half_sen_length):
				# reshape to matrix: [SEQ][BATCH][FRAME]
				self.rnn.train_tbptt(j, j+sentence_length)

			if DEBUG:
				err = self.test_model(t_in, t_l)[0]
				e_time = time.clock()
				print "Error rate in epoch %s, is %.3f. Training time so far is: %.2fm" % ( i+SINDEX, err, (e_time-s_time) / 60.)
				# print ''.join([self.ndict.gettoken(x) for x in r_labels])

			if SAVE:
				class_name = self.__class__.__name__
				self.savemodel("./data/%s/%s.model.epoch%s.n_hidden%s.ssl%s.truncstep%s.dr%s.obj" % (class_name, class_name, i+SINDEX, self.n_hidden, sen_slice_length, self.truncate_step, self.dropout))

			if lr_coef > 0:
				# update learning_rate
				lr = self.rnn.lr.get_value() * lr_coef
				self.rnn.lr.set_value(numpy.array(lr, dtype=theano.config.floatX))

		e_time = time.clock()
		print "RnnLM train over!! The total training time is %.2fm." % ((e_time - s_time) / 60.) 

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
