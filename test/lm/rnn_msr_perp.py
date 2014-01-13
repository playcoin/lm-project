#-*- coding: utf-8 -*-
'''
Created on 2013-05-05 23:00
@summary: Test case on RnnLM
@author: Playcoin
'''

from nlpdict import NlpDict
from pylm import RnnLM
from pylm import RnnEmbTrLM
from fileutil import readClearFile
import numpy
import time
import theano.sandbox.cuda


train_text = readClearFile("./data/datasets/msr_lm_train.ltxt")
nlpdict = NlpDict(comb=False, combzh=False, text=train_text)
print "NlpDict size is:", nlpdict.size()

rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=350, lr=0.5, batch_size=150, truncate_step=4, 
	train_emb=False, dr_rate=0.0,
	backup_file_path="./data/model/RnnEmbTrLM.model.epoch60.n_hidden350.ssl20.truncstep4.dr0.0.embsize5127.in_size5127.rc942.MSR.obj"
)


# rnnlm = RnnEmbTrLM(nlpdict, n_hidden=1200, lr=0.5, batch_size=120, truncate_step=5, 
# 		train_emb=True, dropout=True,
# 		backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch30.n_hidden1200.ssl20.truncstep5.drTrue.embsize100.in_size4702.obj"
# 	)
# rnnlm = RnnEmbTrLM(nlpdict, 
# 		n_emb=200,
# 		n_hidden=400, 
# 		lr=0.5, 
# 		batch_size=150, 
# 		truncate_step=4, 
# 		train_emb=True,
# 		dr_rate=0.,
# 		backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch50.n_hidden400.ssl20.truncstep4.drFalse.embsize200.in_size5127.r7ge200.c935.MSR.obj"
# 	)
# rnnlm = RnnEmbTrLM(nlpdict, 
# 		n_emb=nlpdict.size(),
# 		n_hidden=150, 
# 		lr=0.5, 
# 		batch_size=150, 
# 		truncate_step=4, 
# 		train_emb=False,
# 		dropout=False,
# 		backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch50.n_hidden150.ssl20.truncstep4.drFalse.embsize5127.in_size5127.rnoemb.MSR.obj"
# 	)
tt = readClearFile("./data/datasets/msr_lm_test.ltxt")

len_tt = len(tt)

s_time = time.clock()
ce = rnnlm.crossentropy(tt)
ceo = ce * len_tt / (len_tt-61)
# ceo = ce * len_tt / (len_tt-51)
ppl = numpy.exp(ce)
pplo = numpy.exp(ceo)
# rankinfo = rnnlm.logaverank(tt)
e_time = time.clock()

print "PPL= %.6f, PPL(OOV)= %.6f, time cost for %s tokens is %.3fs" % (ppl, pplo, len_tt, (e_time - s_time))
# print "Avelogrank= %.6f, rank1wSent= %.6f, rank5wSent= %.6f, rank10wSent= %.6f" % rankinfo