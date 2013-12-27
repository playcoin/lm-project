# -*- coding: utf-8 -*-
'''
Created on 2013-06-12 14:28
@summary: 
@author: Playcoin
'''

from nlpdict import NlpDict
from pylm import MlpNgram
from fileutil import readClearFile
import numpy
import time
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')


train_text = readClearFile("./data/datasets/msr_ws_train.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)
print "NlpDict size is:", nlpdict.size()
tt = readClearFile("./data/datasets/msr_ws_test.ltxt")

mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=1200, lr=0.5, batch_size=200, dropout=False,
		backup_file_path='./data/MlpNgram/Mlp7gram.model.epoch50.n_hidden1200.drTrue.n_emb200.in_size5086.rTrue.MSR.obj')

# mlp_ngram = MlpNgram(nlpdict, N=8, n_emb=50, n_hidden=200, lr=0.5, batch_size=200, dropout=False,
# 		backup_file_path='./data/MlpNgram/Mlp8gram.model.epoch20.n_hidden200.obj',
# 		emb_file_path="./data/pku_embedding.obj")

s_time = time.clock()
ce = mlp_ngram.crossentropy(tt)
# ceo = ce * len(tt) / (len(tt)-51)
ceo = ce * len(tt) / (len(tt)-61)
ppl = numpy.exp(ce)
pplo = numpy.exp(ceo)
rankinfo = mlp_ngram.logaverank(tt)
e_time = time.clock()
# 
print "PPL= %.6f, PPL(OOV)= %.6f, time cost for %s tokens is %.3fs" % (ppl, pplo, len(tt), (e_time - s_time))
print "Avelogrank= %.6f, rank1wSent= %.6f, rank5wSent= %.6f, rank10wSent= %.6f" % rankinfo