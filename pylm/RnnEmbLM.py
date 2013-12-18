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


class RnnEmbTrLM(RnnLM):
	'''
	@summary: Rnn language Model use Character Embedding, and ajust embedding in the training time.
	'''

	def __init__(self, ndict, n_emb, n_hidden, lr, batch_size, l2_reg=0.000001, train_emb=True, emb_file_path = None, dropout=False, dr_rate=0.5, truncate_step=5, backup_file_path=None):

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
		self.dr_rate = dr_rate
		self.train_emb = train_emb
		self.in_size = self.out_size  = ndict.size()
		self.n_emb = n_emb
		self.l2_reg = l2_reg

		if emb_file_path:
			f = open(emb_file_path)
			self.embvalues = cPickle.load(f)
			f.close()
			self.embvalues = self.embvalues.astype(theano.config.floatX)
			
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
				self.n_emb, 
				self.n_hidden, 
				self.out_size,
				self.batch_size,
				self.lr,
				self.dropout,
				params = self.rnnparams,
				embeddings = self.embvalues,
				dr_rate = self.dr_rate
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

	def tids2nndata(self, tidseq, truncate_input = True, shared=True):
		# print tidseq.shape
		seq_size = len(tidseq)
		if truncate_input:
			seq_size -= 1

		# for CUDA
		vec_in = theano.shared(numpy.asarray(tidseq[:seq_size], dtype="int32"), borrow=True)
		vec_out = theano.shared(numpy.asarray(tidseq[1:], dtype="int32"), borrow=True)

		if shared:
			return vec_in, vec_out
		else:
			return vec_in.get_value(borrow=True), vec_out.get_value(borrow=True)

	def reshape(self, in_data, l_data):
		s_in = in_data.reshape(self.batch_size, in_data.shape[0] / self.batch_size).T
		s_l = l_data.reshape(self.batch_size,  l_data.shape[0] / self.batch_size).T
		return theano.shared(s_in, borrow=True), theano.shared(s_l, borrow=True)

	def traintext(self, text, test_text, 
			add_se=True, sen_slice_length=4, epoch=200, lr_coef = -1., 
			DEBUG = False, SAVE=False, SINDEX=1, r_init=False):

		self.initRnn()

		tidseq = self.tokens2ids(text, add_se)
		seq_size = len(tidseq)

		# variables for slice sentence
		sentence_length = sen_slice_length * self.truncate_step
		half_sen_length = sentence_length / 2
		data_slice_size = sentence_length * self.batch_size
		data_size = seq_size / data_slice_size * data_slice_size

		mat_in_total, label_total = self.tids2nndata(tidseq, shared=False)
		re_mat_in, re_label = self.reshape(mat_in_total[:data_size], label_total[:data_size])
		self.rnn.build_tbptt(re_mat_in, re_label, self.truncate_step, self.train_emb, l2_reg=self.l2_reg)
		print "Compile Truncate-BPTT Algorithm complete!"

		ttids = self.tokens2ids(test_text, add_se)
		t_in, t_l = self.tids2nndata(ttids, shared=False)

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
				print "Error rate in epoch %s, is %.5f. Training time so far is: %.2fm" % ( i+SINDEX, err, (e_time-s_time) / 60.)
				# print ''.join([self.ndict.gettoken(x) for x in r_labels])

			if SAVE:
				class_name = self.__class__.__name__
				self.savemodel("./data/%s/%s.model.epoch%s.n_hidden%s.ssl%s.truncstep%s.dr%s.embsize%s.in_size%s.r%s.obj" % (class_name, class_name, i+SINDEX, self.n_hidden, sen_slice_length, self.truncate_step, self.dropout, self.n_emb, self.in_size, r_init))

			if lr_coef > 0:
				# update learning_rate
				lr = self.rnn.lr.get_value() * lr_coef
				self.rnn.lr.set_value(numpy.array(lr, dtype=theano.config.floatX))

		e_time = time.clock()
		print "RnnLM train over!! The total training time is %.2fm." % ((e_time - s_time) / 60.) 

	def savemodel(self, filepath):
		backupfile = open(filepath, 'w')
		dumpdata = [self.batch_size, self.n_hidden, self.lr, self.truncate_step, self.rnnparams]
		if self.rnn.C:
			dumpdata.append(self.rnn.C.get_value())

		cPickle.dump(dumpdata, backupfile)
		backupfile.close()
		print "Save model complete! Filepath:", filepath

	def loadmodel(self, filepath):
		backupfile = open(filepath)
		dumpdata = cPickle.load(backupfile)
		self.batch_size, self.n_hidden, self.lr, self.truncate_step, self.rnnparams = dumpdata[:5]
		if len(dumpdata) > 5:
			self.embvalues = dumpdata[5]
		else:
			self.embvalues = None

		backupfile.close()
		print "Load model complete! Filepath:", filepath

	def dumpembeddings(self, filepath):
		backupfile = open(filepath, 'w')
		if self.rnn:
			dumpdata = self.rnn.C.get_value()
		else:
			dumpdata = self.embvalues

		cPickle.dump(dumpdata, backupfile)
		backupfile.close()
		print "Dump embeddings complete! Filepath:", filepath
