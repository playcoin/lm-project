# -*- coding: utf-8 -*-
'''
Created on 2013-06-06 12:43
@summary: 
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.RnnEmbLM import RnnEmbTrLM
import numpy
import time
import theano.sandbox.cuda

#############
# Trainging #
#############
# text
# f = file('./data/msr_training.ltxt')
f = file('./data/pku_train_nw.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

train_text = text
test_text = text[:40000]
len_text = len(train_text)

nlpdict = NlpDict()
nlpdict.buildfromtext(train_text, freq_thres=0)
# nlpdict.transEmbedding('./data/pku_closed_word_embedding100.ltxt', "./data/pku_embedding_rnnembtr100_c1.obj")
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

# use gpu
theano.sandbox.cuda.use('gpu1')
rnnlm = RnnEmbTrLM(nlpdict, n_hidden=1200, lr=0.5, batch_size=150, truncate_step=5, 
	train_emb=True, dropout=True,
	emb_file_path="./data/pku_embedding_rnnembtr100.obj")

# # use gpu
# theano.sandbox.cuda.use('gpu0')
# rnnlm = RnnEmbTrLM(nlpdict, n_hidden=1200, lr=0.3, batch_size=120, truncate_step=6, 
# 	train_emb=True, dropout=True,
# 	emb_file_path="./data/pku_embedding_rnnembtr100.obj")

# rnnlm = RnnEmbTrLM(nlpdict, n_hidden=1000, lr=0.6, batch_size=120, truncate_step=4, 
# 	train_emb=True, dropout=True,
# 	backup_file_path="./data/MSR/RnnEmbTrLM/RnnEmbTrLM.model.epoch61.n_hidden1000.ssl20.truncstep4.drTrue.embsize100.obj")
# rnnlm.lr = 0.6 * 0.96 ** 61

rnnlm.traintext(train_text, test_text, 
		add_se=False, 
		sen_slice_length=20, 
		epoch=100, 
		lr_coef=0.96, 
		DEBUG=True,
		SAVE=True,
		SINDEX=1
	)
