# -*- coding: utf-8 -*-
'''
Created on 2013-06-06 12:43
@summary: 
@author: Playcoin
'''

from nlpdict import NlpDict
from pylm import RnnEmbTrLM
from pylm import MlpNgram
from fileutil import readClearFile

import numpy
import time
import theano.sandbox.cuda
# use gpu

#############
# Trainging #
#############
# text
train_text = readClearFile("./data/datasets/pku_lm_train.ltxt")
nlpdict = NlpDict(comb=True, combzh=False, text=train_text)

valid_text = readClearFile("./data/datasets/pku_lm_valid.ltxt")

test_text = train_text[:5001]
len_text = len(train_text)

print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

# training case 4
theano.sandbox.cuda.use('gpu1')
rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5,
	emb_file_path="./data/7gram.emb200.h1200.d4566.emb.obj"
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)