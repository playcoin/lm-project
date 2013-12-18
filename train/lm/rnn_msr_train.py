# -*- coding: utf-8 -*-
'''
Created on 2013-06-13 21:21
@summary: 
@author: Playcoin
'''

from nlpdict import NlpDict
from pylm import RnnEmbTrLM
from pylm import MlpNgram
import numpy
import time
from threading import Timer
import theano.sandbox.cuda

#############
# Trainging #
#############
# text
f = file('./data/msr_train.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

train_text = text
test_text = text[:501]
len_text = len(train_text)

nlpdict = NlpDict()
nlpdict.buildfromtext(train_text, freq_thres=0)
# nlpdict.transEmbedding('./data/pku_closed_word_embedding100_n.ltxt', "./data/sohu_embedding_rnnembtr100.obj")
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

# use gpu
theano.sandbox.cuda.use('gpu1')

# mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=200, n_hidden=1200, lr=0.5, batch_size=200, dropout=False,
# 		backup_file_path='./data/MlpNgram/Mlp7gram.model.epoch90.n_hidden1200.drFalse.n_emb200.in_size5127.rTrue.MSR.obj')
# mlp_ngram.dumpembedding('./data/7gram.emb200.h1200.d5172.emb.obj')

rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1000, lr=0.5, batch_size=150, truncate_step=4, 
	train_emb=True, dropout=True,
	# emb_file_path="./data/7gram.emb200.h1200.d5172.emb.obj")
	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch41.n_hidden1000.ssl20.truncstep4.drTrue.embsize200.in_size5127.r7ge200.c94.MSR.obj")

# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.5, batch_size=150, truncate_step=4, 
# 	train_emb=False, dropout=False,
# )
rnnlm.lr = 0.5 * 0.94 ** 41
rnnlm.traintext(train_text, test_text, 
		add_se=False, sen_slice_length=20, epoch=9, lr_coef=0.94, 
		DEBUG=True, SAVE=True, SINDEX=42, r_init="7ge200.c94.MSR"
	)
