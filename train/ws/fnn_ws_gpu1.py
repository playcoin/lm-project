# -*- coding: utf-8 -*-
'''
Created on 2013-12-02 15:35
@summary: FNN 中文测试
@author: egg
'''

from nlpdict import NlpDict
from pyws import MlpWS
from fileutil import readClearFile, writeFile

import numpy
import time
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')

#############
# Data file #
#############
train_text = readClearFile("./data/datasets/msr_ws_train.ltxt")
train_tags = readClearFile("./data/datasets/msr_ws_train_tag.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

valid_text = readClearFile("./data/datasets/msr_ws_valid.ltxt")
valid_tags = readClearFile("./data/datasets/msr_ws_valid_tag.ltxt")

test_text = readClearFile("./data/datasets/msr_ws_test.ltxt")
test_tags = readClearFile("./data/datasets/msr_ws_test_tag.ltxt")

train_text = train_text + "\n" + valid_text
train_tags = train_tags + "\n" + valid_tags

print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))

mlp_ws = MlpWS(nlpdict, chunk_size=5, n_emb=200, n_hidden=600, lr=0.5, batch_size=200, dropout=False,
		emb_file_path='data/7gram.emb200.h1200.d5086.emb.obj')

mlp_ws.traintext(train_text, train_tags, test_text[:5000], test_tags[:5000], 
	DEBUG=1, SAVE=5, SINDEX=1, epoch=70, lr_coef=0.94, r_init="7g200")


mlp_ws = MlpWS(nlpdict, chunk_size=5, n_emb=200, n_hidden=1200, lr=0.5, batch_size=200, dropout=True,
		emb_file_path="data/7gram.emb200.h1200.d5086.emb.obj")

mlp_ws.traintext(train_text, train_tags, test_text[:5000], test_tags[:5000], 
	DEBUG=1, SAVE=5, SINDEX=1, epoch=70, lr_coef=0.94, r_init="7g200")


mlp_ws = MlpWS(nlpdict, chunk_size=5, n_emb=200, n_hidden=600, lr=0.5, batch_size=200, dropout=False,
		emb_file_path=None)

mlp_ws.traintext(train_text, train_tags, test_text[:5000], test_tags[:5000], 
	DEBUG=1, SAVE=5, SINDEX=1, epoch=70, lr_coef=0.94)