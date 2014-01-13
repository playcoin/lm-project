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
# theano.sandbox.cuda.use('gpu1')

train_text = readClearFile("./data/datasets/pku_lm_train.ltxt")
nlpdict = NlpDict(comb=False, combzh=False, text=train_text)
print "NlpDict size is:", nlpdict.size()

rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=200, lr=0.5, batch_size=150, truncate_step=4, 
		train_emb=False, dr_rate=0.0,
		backup_file_path="./data/model/RnnEmbTrLM.model.epoch100.n_hidden200.ssl20.truncstep4.dr0.0.embsize4633.in_size4633.rc94.obj"
	)

# rnnlm = RnnEmbTrLM(nlpdict, 
# 		n_emb=4633,
# 		n_hidden=150, 
# 		lr=0.5, 
# 		batch_size=150, 
# 		truncate_step=4, 
# 		train_emb=False,
# 		dropout=False,
# 		backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch100.n_hidden150.ssl20.truncstep4.drFalse.embsize4633.in_size4633.rTrue.obj"
# 	)

# tt = u"天津市首届检察官艺术节音乐会日前举行。图为天津市检察官大合唱《检察官之歌》。（新华社记者李昌元摄）"

# probs = rnnlm.likelihood(tt, debug = True)

# f = file('./data/pku_valid.ltxt')
# f = file('./data/pku_test.ltxt')
tt = readClearFile("./data/datasets/pku_lm_test.ltxt")

s_time = time.clock()
ce = rnnlm.crossentropy(tt)
# ceo = ce * 149924 / 149825
# ceo = ce * 174677 / 174305
ceo = ce * 174677 / 174297
ppl = numpy.exp(ce)
pplo = numpy.exp(ceo)
# rankinfo = rnnlm.logaverank(tt)
e_time = time.clock()

print "PPL= %.6f, PPL(OOV)= %.6f, time cost for %s tokens is %.3fs" % (ppl, pplo, len(tt), (e_time - s_time))
# print "Avelogrank= %.6f, rank1wSent= %.6f, rank5wSent= %.6f, rank10wSent= %.6f" % rankinfo