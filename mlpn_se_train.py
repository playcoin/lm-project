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

# use gpu
theano.sandbox.cuda.use('gpu1')
mlp_ngram = MlpNgram(nlpdict, N=7, n_emb=100, n_hidden=1200, lr=0.5, batch_size=300, dropout=True,
		emb_file_path='./data/pku_embedding_100.obj')

# mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=50, n_hidden=1000, lr=0.5, batch_size=150, dropout=True,
# 		backup_file_path='./data/MlpNgram/Mlp5gram.model.epoch63.n_hidden1000.drTrue.in_size4702.obj')

# mlp_ngram.lr = 0.5 * 0.96 ** 63


print "Train size is: %s, testing size is: %s" % (len(train_text), len(test_text))
mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=1, epoch=100, lr_coef=0.96)

# ce = mlp_ngram.crossentropy(test_text)
# print "Cross-entropy is:", ce
# print "Perplexity is:", numpy.exp(ce)

# print "Log rank is:", mlp_ngram.logaverank(test_text) 

# s_prefix = u"拥挤"
# top_tids, top_probs = mlp_ngram.topN(s_prefix, 10)
# for x in top_tids:
# 	print nlpdict.gettoken(x),
# print 
# print top_probs