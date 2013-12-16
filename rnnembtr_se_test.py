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
# use gpu
theano.sandbox.cuda.use('gpu0')

#############
# Trainging #
#############
# text
f = file('./data/datasets/pku_train_large.ltxt')
text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

nlpdict = NlpDict(comb=True)
nlpdict.buildfromtext(text)

f = file('./data/datasets/pku_valid_small.ltxt')
valid_text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

train_text = text + '\n' + valid_text
test_text = train_text[:5001]
len_text = len(train_text)

# nlpdict.transEmbedding('./data/pku_closed_word_embedding100.ltxt', "./data/pku_embedding_rnnembtr100_c1.obj")
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

# training case 4
rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True,
	emb_file_path="./data/7gram.emb200.h1200.d4598.emb.obj"
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=True, SAVE=True, SINDEX=1, r_init="c94"
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
