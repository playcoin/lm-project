#-*- coding: utf-8 -*-
'''
Created on 2013-05-05 23:00
@summary: Test case on RnnLM
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.RnnLM import RnnLM
from pylm.RnnEmbLM import RnnEmbLM, RnnEmbTrLM
import numpy
import time
import theano.sandbox.cuda


train_file_path = './data/sohu_train.ltxt'
print "Training file path:", train_file_path
# text
f = file(train_file_path)
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

#############
# Trainging #
#############
train_text = text

nlpdict = NlpDict()
nlpdict.buildfromtext(train_text, freq_thres=0)
print "NlpDict size is:", nlpdict.size()


theano.sandbox.cuda.use('gpu1')
# rnnlm = RnnEmbTrLM(nlpdict, n_hidden=1200, lr=0.5, batch_size=120, truncate_step=5, 
# 		train_emb=True, dropout=True,
# 		backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch30.n_hidden1200.ssl20.truncstep5.drTrue.embsize100.in_size4702.obj"
# 	)
rnnlm = RnnEmbTrLM(nlpdict, 
		n_hidden=1200, 
		lr=0.5, 
		batch_size=120, 
		truncate_step=4, 
		train_emb=True,
		dropout=True,
		backup_file_path="./data/SOHU/RnnEmbTrLM/RnnEmbTrLM.model.epoch68.n_hidden1200.ssl20.truncstep4.drTrue.embsize100.in_size5005.obj"
	)


f = file('./data/sohu_valid.ltxt')
# f = file('./data/sohu_test.ltxt')
tt = unicode(f.read(), 'utf-8')
f.close()

len_tt = len(tt)

s_time = time.clock()
ce = rnnlm.crossentropy(tt)
ceo = ce * len_tt / (len_tt-363)
# ceo = ce * len_tt / (len_tt-118)
ppl = numpy.exp(ce)
pplo = numpy.exp(ceo)
# rankinfo = rnnlm.logaverank(tt)
e_time = time.clock()

print "PPL= %.6f, PPL(OOV)= %.6f, time cost for %s tokens is %.3fs" % (ppl, pplo, len(tt), (e_time - s_time))
# print "Avelogrank= %.6f, rank1wSent= %.6f, rank5wSent= %.6f, rank10wSent= %.6f" % rankinfo