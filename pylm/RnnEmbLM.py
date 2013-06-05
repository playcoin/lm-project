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

from RnnLM import RnnLM
from dltools.rnn import RNN

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

