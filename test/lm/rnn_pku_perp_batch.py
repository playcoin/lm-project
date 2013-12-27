#-*- coding: utf-8 -*-
'''
Created on 2013-05-05 23:00
@summary: Test case on RnnLM
@author: Playcoin
'''

from nlpdict import NlpDict
from pylm import RnnLM
from pylm import RnnEmbTrLM
import numpy
import time
import theano.sandbox.cuda


train_text = readClearFile("./data/datasets/msr_ws_train.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)
print "NlpDict size is:", nlpdict.size()


theano.sandbox.cuda.use('gpu1')
# f = file('./data/pku_valid.ltxt')
f = file('./data/datasets/pku_lm_test.ltxt')
tt = unicode(f.read(), 'utf-8')
f.close()


def test(lm):
	s_time = time.clock()
	ce = lm.crossentropy(tt)
	# ceo = ce * 149924 / 149825
	ceo = ce * 174677 / 174297
	# ppl = numpy.exp(ce)                        
	pplo = numpy.exp(ceo)
	e_time = time.clock()

	print "PPL(OOV)= %.6f, time cost for %s tokens is %.3fs" % (pplo, len(tt), (e_time - s_time))
	return pplo

epoches = [1, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100]
# epoches = [5, 8, 11, 14]#, 17, 20]
# epoches = [1, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50]
# epoches = [30, 40]                                                                     

epoches.reverse()
ppls = []
for i in epoches:
	modelpath = "./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch%d.n_hidden1200.ssl20.truncstep4.dr0.5.embsize200.in_size4633.rbs150.c96.obj" % i
	rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=150, truncate_step=4, 
			train_emb=True, dr_rate=0.5,
			backup_file_path=modelpath
		)

	ppls.append(test(rnnlm))

print ppls
