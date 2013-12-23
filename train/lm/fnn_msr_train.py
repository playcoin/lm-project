# -*- coding: utf-8 -*-
'''
Created on 2013-06-24 15:53
@summary: test case for MlpNgram in MSR
@author: egg
'''

from nlpdict import NlpDict
from pylm import MlpNgram
from fileutil import readClearFile
import numpy
import time
import theano.sandbox.cuda

#############
# Trainging #
#############
# text
train_text = readClearFile("./data/datasets/msr_lm_train.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

test_text = train_text[:5001]
print "Dict size is: %s, Training size is: %s" % (nlpdict.size(), len(train_text))

# use gpu
theano.sandbox.cuda.use('gpu0')
# test random init
mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=1200, lr=0.6, batch_size=200, 
	dropout=True, emb_file_path=None)

mlp_ngram.traintext(train_text, test_text, 
	DEBUG=True, SAVE=True, SINDEX=1, r_init="True.MSR",
	epoch=50, lr_coef=0.91)