# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpNgram
@author: egg
'''

from nlpdict import NlpDict
from pylm import MlpNgram
import numpy
import time
import theano.sandbox.cuda

#############
# Trainging #
#############
# text
f = file('./data/datasets/pku_train_large_ws.ltxt')
text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

# NlpDict
nlpdict = NlpDict(comb=True)
nlpdict.buildfromtext(text, freq_thres=0)
print "NlpDict size is:", nlpdict.size()
train_text = text
test_text = text[:501]
print "Train size is: %s, testing size is: %s" % (len(train_text), len(test_text))

# use gpu
theano.sandbox.cuda.use('gpu0')
# test random init
# mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=50, n_hidden=800, lr=0.5, batch_size=200, 
# 	dropout=False, emb_file_path=None)

# mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)


# test random init
mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=1200, lr=0.5, batch_size=200, 
	dropout=False, emb_file_path=None)

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.95)

###########
# Dropout #
###########

# Training example 6
# mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=200, n_hidden=2000, lr=0.5, batch_size=200, dropout=True,
# 		emb_file_path='./data/5gram.emb200.h600.d4633.emb.obj')

# mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)

# Training example 7
# mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=200, n_hidden=2400, lr=0.5, batch_size=200, 
# 		l2_reg=0.0000001, dropout=True,
# 		emb_file_path='./data/5gram.emb200.h600.d4633.emb.obj')

# mlp_ngram.traintext(train_text, test_text, epoch=100, lr_coef=0.96, 
# 	DEBUG=True, SAVE=True, SINDEX=1, r_init="e200init")