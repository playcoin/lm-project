# -*- coding: utf-8 -*-
'''
Created on 2013-06-24 15:53
@summary: test case for MlpNgram in MSR
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
# text
f = file('./data/msr_train.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

# NlpDict
nlpdict = NlpDict()
nlpdict.buildfromtext(text, freq_thres=0)
print "NlpDict size is:", nlpdict.size()
# nlpdict.transEmbedding('./data/pku_closed_word_embedding100.ltxt', "./data/pku_embedding_100.obj")
train_text = text
test_text = text[:5001]
print "Train size is: %s, testing size is: %s" % (len(train_text), len(test_text))

# use gpu
theano.sandbox.cuda.use('gpu1')
# test random init
# mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=2000, lr=0.6, batch_size=200, 
# 	dropout=True, emb_file_path=None)

mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=2000, lr=0.6, batch_size=200, 
	dropout=True, emb_file_path=None)

mlp_ngram.traintext(train_text, test_text, 
	DEBUG=True, SAVE=True, SINDEX=1, r_init="True.MSR",
	epoch=50, lr_coef=0.96)
