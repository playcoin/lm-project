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
len_text = len(train_text)

nlpdict = NlpDict()
nlpdict.buildfromtext(train_text)
# nlpdict.transEmbedding('./data/pku_closed_word_embedding100.ltxt', "./data/pku_embedding_rnnembtr100_c1.obj")
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

# use gpu
theano.sandbox.cuda.use('gpu1')


rnnlm = RnnEmbTrLM(nlpdict, n_emb=50, n_hidden=600, lr=0.2, batch_size=150, 
	l2_reg=0.0000001, truncate_step=4, train_emb=True, dropout=True,
	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch44.n_hidden600.ssl20.truncstep4.drTrue.embsize100.in_size4633.r7g100.c95.obj"
)

rnnlm.dumpembeddings("./data/RnnEmbTrLM.n_hidden600.embsize100.in_size4633.emb.obj")