# -*- coding: utf-8 -*-
'''
Created on 2013-06-12 14:28
@summary: 
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.MlpNgram import MlpNgram
import numpy
import time
import theano.sandbox.cuda


train_file_path = './data/msr_train.ltxt'
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
print "Train size is: %s" % len(train_text)

theano.sandbox.cuda.use('gpu1')

# mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=200, n_hidden=800, lr=0.5, batch_size=200, dropout=False,
# 		backup_file_path='./data/MlpNgram/Mlp5gram.model.epoch100.n_hidden1600.drFalse.n_emb200.in_size4633.rTrue.obj')

mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=1200, lr=0.5, batch_size=200, dropout=False,
		backup_file_path='./data/MlpNgram/Mlp7gram.model.epoch90.n_hidden1200.drFalse.n_emb200.in_size5127.rTrue.MSR.obj')

# mlp_ngram = MlpNgram(nlpdict, N=8, n_emb=50, n_hidden=200, lr=0.5, batch_size=200, dropout=False,
# 		backup_file_path='./data/MlpNgram/Mlp8gram.model.epoch20.n_hidden200.obj',
# 		emb_file_path="./data/pku_embedding.obj")

# f = file('./data/msr_valid.ltxt')
f = file('./data/msr_test.ltxt')
tt = unicode(f.read(), 'utf-8')
f.close()

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