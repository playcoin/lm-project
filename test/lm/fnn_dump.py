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
f = file('./data/datasets/pku_train_large.ltxt')
text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

# NlpDict
nlpdict = NlpDict(comb=True)
nlpdict.buildfromtext(text, freq_thres=0)
print "NlpDict size is:", nlpdict.size()

len_text = len(text)
train_text = text

print "Train size is: %s" % len_text

theano.sandbox.cuda.use('gpu0')
mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=1200, lr=0.5, batch_size=200, 
	dropout=False, 
	backup_file_path="./data/MlpNgram/Mlp7gram.model.epoch100.n_hidden1200.drFalse.n_emb200.in_size4598.rTrue.obj")
mlp_ngram.dumpembedding("7gram.emb200.h1200.d4598.emb.obj")


