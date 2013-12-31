# -*- coding: utf-8 -*-
'''
Created on 2013-12-23 11:43
@summary: Dump models for ubuntu
@author: Playcoin
'''
from nlpdict import NlpDict
from pyws import RnnWFWS, RnnWFWS2
from pylm import RnnEmbTrLM
from fileutil import readClearFile

import theano
# theano.sandbox.cuda.use('gpu0')

train_text = readClearFile("./data/datasets/pku_lm_train.ltxt")
nlpdict = NlpDict(comb=True, combzh=False, text=train_text)

rnnws = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=150, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5, #ext_emb=2,
	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch10.n_hidden1200.ssl20.truncstep4.dr0.5.embsize200.in_size4566.rc94.obj"
)
rnnws.initRnn()
# rnnws.savemodel("./data/RnnWFWS2/RnnWFWS2.model.epoch60.n_hidden1400.ssl20.truncstep4.drTrue.embsize200.in_size4598.rtremb.c91.wb.obj")
rnnws.savemodel("data/RnnEmbTrLM/RnnEmbTrLM.model.epoch10.n_hidden1200.ssl20.truncstep4.dr0.5.embsize200.in_size4566.rc94.b.obj")