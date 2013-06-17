# -*- coding: utf-8 -*-
'''
Created on 2013-05-05 16:26
@summary: The Abstract Base LM Class
@author: egg
'''
import numpy
import theano
from abc import ABCMeta, abstractmethod

class LMBase(object):
	'''
	@summary: The Abstract Base LM Class
	'''
	__metaclass__ = ABCMeta

	def __init__(self, ndict):
		'''
		@summary: Construct method, initial NlpDict.
		
		@param ndict:
		'''
		self.ndict = ndict
		

	################
	# Base methods #
	################
	def tokens2ids(self, tokenseq, add_se=False):
		'''
		@summary: Token chars to token ids
		
		@param tokenseq: token chars
		@param add_se: whether add head and tail symbol
		@result: token ids
		'''
		tidseq = add_se and [self.ndict.getstartindex()] or []
		tidseq.extend([self.ndict[token] for token in tokenseq])
		add_se and tidseq.extend([self.ndict.getendindex()])

		return tidseq

	def tids2nndata(self, tidseq, truncate_input = True, shared = True):
		'''
		@summary: token ids to theano function input variables (matrix and vector)
		'''
		# print tidseq.shape
		in_size = len(tidseq)
		if truncate_input:
			in_size -= 1

		mat_in = numpy.zeros((in_size, self.ndict.size()), dtype=theano.config.floatX)
		vec_out = numpy.asarray(tidseq[1:], dtype="int32")

		for i in xrange(in_size):
			mat_in[i][tidseq[i]] = numpy.array(1., dtype=theano.config.floatX)

		if shared:
			mat_in = theano.shared(mat_in, borrow=True)
			vec_out = theano.shared(vec_out, borrow=True)

		return mat_in, vec_out


	####################
	# Abstract methods #
	####################
	@abstractmethod
	def traintext(self, text, add_se=False):
		'''
		@summary: Train model by the input text
		
		@param text: training text
		@param add_se: flag to indicate whether add START and END symbol for input text
		'''
		return

	@abstractmethod
	def testtext(self, text):
		'''
		@summary: Test the model, and return the error rate of given text.
			Generate input and label from text.
		
		@param text: test text
		@result: error rate
		'''
		return

	@abstractmethod
	def likelihood(self, text):
		'''
		@summary: Return the likelihood of the whole sentence.

		@param text: sentence text
		@result: list of each likelihood for each token 
		'''
		return

	@abstractmethod
	def predict(self, text):
		'''
		@summary: Predict the most probable token to follow the given text.
		
		@param text: history text
		@result: the next token id
		'''
		return

	@abstractmethod
	def ranks(self, sentence):
		'''
		@summary: Return the rank list of each char in a sentence. 
		
		@param sentence:
		@result: 
		'''
		return

	def crossentropy(self, text):
		'''
		@summary: Return the cross-entropy of the text.
			Use the equation: H(W) = - 1. / N * (\sum P(w_1 w_2 ... w_N))
		
		@param text:
		@result: cross entropy
		'''
		# slice text to sentences
		sents = text.split('\n')
		log_probs = []
		for sentence in sents:
			if sentence != "":
				log_probs.extend(numpy.log(self.likelihood(sentence)))

		crossentropy = - numpy.mean(log_probs)

		return crossentropy

	def logaverank(self, text):
		'''
		@summary: Return the average log rank
		
		@param text:
		@result: 
		'''
		# slice text to sentences
		sents = text.split('\n')
		rank_list = []
		for sentence in sents:
			if sentence != "":
				rank_list.extend(self.ranks(sentence))

		f_len = float(len(rank_list))
		count1 = len([x for x in rank_list if x == 1])
		count5 = len([x for x in rank_list if x <= 5])
		count10 = len([x for x in rank_list if x <= 10])

		log_ranks = numpy.log(rank_list)
		return numpy.mean(log_ranks), count1 / f_len, count5 / f_len, count10 / f_len 

	@abstractmethod
	def savemodel(self, backup_file_path):
		'''
		@summary: Back up the language model by cPickle
		
		@param backup_file_path:
		'''
		return

	@abstractmethod
	def loadmodel(self, backup_file_path):
		'''
		@summary: Load the language model by cPickle
		
		@param backup_file_path:
		'''
		return


