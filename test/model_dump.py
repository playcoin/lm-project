# -*- coding: utf-8 -*-
'''
Created on 2013-12-23 11:43
@summary: Dump models for ubuntu
@author: Playcoin
'''
from nlpdict import NlpDict
from pyws import RnnWFWS, RnnWFWS2
from fileutil import readClearFile

import theano
theano.sandbox.cuda.use('gpu0')

train_text = readClearFile("./data/datasets/pku_ws_train_large.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

rnnws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=150, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True, #ext_emb=2,
	backup_file_path="./data/RnnWFWS2.model.epoch60.n_hidden1400.ssl20.truncstep4.drTrue.embsize200.in_size4598.rtremb.c91.obj"
)
rnnws.initRnn()
# rnnws.savemodel("./data/RnnWFWS2/RnnWFWS2.model.epoch60.n_hidden1400.ssl20.truncstep4.drTrue.embsize200.in_size4598.rtremb.c91.wb.obj")
rnnws.dumpembedding("data/RnnWFWS2.n_hidden1400.embsize200.in_size4598.embeddings.obj")