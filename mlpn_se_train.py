# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpNgram
@author: egg
'''

from nlpdict.NlpDict import NlpDict
from pylm.MlpNgram import MlpNgram
import numpy
import time
import theano.sandbox.cuda

#############
# Trainging #
#############
# text
f = file('./data/pku_train_nw.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

# NlpDict
nlpdict = NlpDict()
nlpdict.buildfromtext(text, freq_thres=0)
print "NlpDict size is:", nlpdict.size()
# nlpdict.transEmbedding('./data/pku_closed_word_embedding100.ltxt', "./data/pku_embedding_100.obj")
train_text = text
test_text = text[:40001]
print "Train size is: %s, testing size is: %s" % (len(train_text), len(test_text))

# use gpu
theano.sandbox.cuda.use('gpu1')
# Training example 1
mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=100, n_hidden=500, lr=0.5, batch_size=200, dropout=False,
		emb_file_path='./data/pku_embedding_100.obj')

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)

# Training example 2
mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=100, n_hidden=600, lr=0.5, batch_size=200, dropout=False,
		emb_file_path='./data/pku_embedding_100.obj')

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)

# Training example 3
mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=100, n_hidden=700, lr=0.5, batch_size=200, dropout=False,
		emb_file_path='./data/pku_embedding_100.obj')

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)

# Training example 4
mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=100, n_hidden=800, lr=0.5, batch_size=200, dropout=False,
		emb_file_path='./data/pku_embedding_100.obj')

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)

###########
# Dropout #
###########
# Training example 5
mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=100, n_hidden=800, lr=0.5, batch_size=200, dropout=True,
		emb_file_path='./data/pku_embedding_100.obj')

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)

# Training example 6
mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=100, n_hidden=1000, lr=0.5, batch_size=200, dropout=True,
		emb_file_path='./data/pku_embedding_100.obj')

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)

# Training example 7
mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=100, n_hidden=1200, lr=0.5, batch_size=200, dropout=True,
		emb_file_path='./data/pku_embedding_100.obj')

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)