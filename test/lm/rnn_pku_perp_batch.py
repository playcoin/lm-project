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


train_file_path = './data/datasets/pku_train.ltxt'
print "Training file path:", train_file_path
# text
f = file(train_file_path)
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

train_text = text

nlpdict = NlpDict()
nlpdict.buildfromtext(train_text, freq_thres=0)
print "NlpDict size is:", nlpdict.size()
print "Train size is: %s" % len(train_text)


theano.sandbox.cuda.use('gpu1')
# f = file('./data/pku_valid.ltxt')
f = file('./data/datasets/pku_test.ltxt')
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
                  
                                                                                                                                                                                                                                                                                             
ppls = []
for i in epoches:
	modelpath = "./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch%d.n_hidden1200.ssl20.truncstep4.drTrue.embsize200.in_size4633.rdr30.c945.obj" % i
	rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=400, lr=0.5, batch_size=150, truncate_step=4, 
			train_emb=True, dropout=False, dr_rate=0.3,
			backup_file_path=modelpath
		)

	ppls.append(test(rnnlm))

print ppls
