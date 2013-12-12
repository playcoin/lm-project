# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpNgram
@author: egg
'''

from nlpdict.NlpDict import NlpDict
from pylm.MlpNgram import MlpNgram
import numpy
import time
import theano.sandbox.cuda

#############
# Trainging #
#############
nlpdict = NlpDict(comb=True)
nlpdict.buildfromfile('./data/pku_train.ltxt')
# text
f = file('./data/pku_train_nw.ltxt')
text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

len_text = len(text)
train_text = text

print "Train size is: %s" % len_text

# theano.sandbox.cuda.use('gpu0')

# mlp_ngram = MlpNgram(nlpdict, hvalue_file="./data/pku_embedding.obj", backup_file_path="./data/MlpNgram/Mlp8gram.model.epoch20.n_hidden200.obj")
# mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=False)
mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=1200, lr=0.5, batch_size=200, 
	dropout=False, 
	backup_file_path="./data/MlpNgram/Mlp7gram.model.epoch100.n_hidden1200.drFalse.n_emb200.in_size4566.rTrue.obj")
mlp_ngram.dumpembedding("7gram.emb200.h1200.d4566.emb.obj")

#############
#  Testing  #
#############

# f = file('./data/pku_test.txt')
# tt = unicode(f.read(), 'utf-8')
# f.close()

# test_text = tt[:100000]

# print "Error rate is:", mlp_ngram.testtext(test_text)

# ce = mlp_ngram.crossentropy(test_text)
# print "Cross-entropy is:", ce
# print "Perplexity is:", numpy.exp(ce)

# print "Log rank is:", mlp_ngram.logaverank(test_text[:50000]) 
