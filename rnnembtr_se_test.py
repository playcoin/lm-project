# -*- coding: utf-8 -*-
'''
Created on 2013-06-06 12:43
@summary: 
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.RnnEmbLM import RnnEmbTrLM
from pylm.MlpNgram import MlpNgram
import numpy
import time
import theano.sandbox.cuda

#############
# Trainging #
#############
# text
# f = file('./data/msr_training.ltxt')
f = file('./data/pku_train.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

train_text = text
test_text = text[:5001]
len_text = len(train_text)

nlpdict = NlpDict()
nlpdict.buildfromtext(train_text)
# nlpdict.transEmbedding('./data/pku_closed_word_embedding100.ltxt', "./data/pku_embedding_rnnembtr100_c1.obj")
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

# use gpu
theano.sandbox.cuda.use('gpu0')

# training case 4
rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=400, lr=0.15, batch_size=150, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=False,
	emb_file_path="./data/7gram.emb200.h1200.d4633.emb.obj"
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=50, lr_coef=0.925, 
	DEBUG=True, SAVE=True, SINDEX=1, r_init="7g200.c925"
)
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.2, batch_size=150, 
# 	l2_reg=0.0000001, truncate_step=4, train_emb=True, dropout=False,
# 	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch20.n_hidden300.ssl20.truncstep4.drFalse.embsize4633.in_size4633.rTrue.obj"
# )
# rnnlm.lr = 0.2 * 0.96 ** 20

# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=600, lr=0.5, batch_size=150, truncate_step=4, 
# 		train_emb=True, dropout=True,
# 		backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch26.n_hidden600.ssl20.truncstep4.drTrue.embsize4633.in_size4633.rnoemb.obj"
# 	)

# rnnlm.lr = 0.5 * 0.96 ** 26
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=74, lr_coef=0.96, 
# 	DEBUG=True, SAVE=True, SINDEX=27, r_init="noemb"
# )

# mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=400, n_hidden=600, lr=0.5, batch_size=200, dropout=False,
# 		backup_file_path='./data/MlpNgram/Mlp5gram.model.epoch100.n_hidden600.drFalse.n_emb400.in_size4633.rTrue.obj')
# mlp_ngram.dumpembedding('./data/5gram.emb400.h600.d4633.emb.obj')

# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dropout=False,
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
# 	DEBUG=True, SAVE=True, SINDEX=1, r_init="noemb"
# )