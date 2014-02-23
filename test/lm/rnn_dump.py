# -*- coding: utf-8 -*-
'''
Created on 2013-06-06 12:43
@summary: 
@author: Playcoin
'''

from nlpdict import NlpDict
from pylm import RnnEmbTrLM
from pylm import MlpNgram
import numpy
import time
import theano.sandbox.cuda
from fileutil import readClearFile, writeFile
#############
# Trainging #
#############
train_text = readClearFile("./data/datasets/pku_ws_train_large.ltxt")
# train_text = readClearFile("./data/datasets/msr_ws_train.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)
# nlpdict.transEmbedding('./data/pku_closed_word_embedding100.ltxt', "./data/pku_embedding_rnnembtr100_c1.obj")
print "Dict size is: %s" % nlpdict.size()

# use gpu

rnnlm = RnnEmbTrLM(nlpdict, n_emb=50, n_hidden=600, lr=0.2, batch_size=150, 
	l2_reg=0.0000001, truncate_step=4, train_emb=True, dr_rate=0.5,
	backup_file_path="./data/RnnEmbTrLM.model.epoch100.n_hidden1200.ssl20.truncstep4.drTrue.embsize200.in_size4598.rc94.obj"
)

rnnlm.dumpembeddings("./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4598.embeddings.obj")