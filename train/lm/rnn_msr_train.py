# -*- coding: utf-8 -*-
'''
Created on 2013-06-13 21:21
@summary: 
@author: Playcoin
'''

from nlpdict import NlpDict
from pylm import RnnEmbTrLM
from pylm import MlpNgram
from fileutil import readClearFile

import numpy
import time
from threading import Timer
import theano.sandbox.cuda

#############
# Trainging #
#############
# text
train_text = readClearFile("./data/datasets/msr_ws_train.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

test_text = train_text[:5001]
print "Dict size is: %s, Training size is: %s" % (nlpdict.size(), len(train_text))
# use gpu
theano.sandbox.cuda.use('gpu0')



# mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=1200, lr=0.5, batch_size=200, dropout=False,
# 		backup_file_path='./data/MlpNgram/Mlp7gram.model.epoch50.n_hidden1200.drTrue.n_emb200.in_size5086.rTrue.MSR.obj')
# mlp_ngram.dumpembedding('./data/7gram.emb200.h1200.d5086.emb.obj')

rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=150, truncate_step=4, 
	train_emb=True, dr_rate=0.5,
	emb_file_path="./data/7gram.emb200.h1200.d5086.emb.obj"
)

rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=50, lr_coef=0.93, 
	DEBUG=True, SAVE=True, SINDEX=1, r_init="7ge200.c93.MSR"
)
