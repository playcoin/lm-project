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
nlpdict.transEmbedding('./data/pku_closed_word_embedding.ltxt', "./data/pku_embedding_c1.obj")

# use gpu
theano.sandbox.cuda.use('gpu0')

# mlp_ngram = MlpNgram(nlpdict, N=4, n_in=50, n_hidden=150, lr=0.07, batch_size=50, hvalue_file="./data/MlpBigram.hiddens.obj")
# mlp_ngram = MlpNgram(nlpdict, N=5, n_in=50, n_hidden=200, lr=0.07, batch_size=50, hvalue_file="./data/pku_embedding_c1.obj")
mlp_ngram = MlpNgram(nlpdict, backup_file_path="./data/MlpNgram/Mlp4gram.model.epoch20.n_hidden150.obj", hvalue_file="./data/pku_embedding.obj")
# mlp_ngram.lr = 0.05

train_text = text[:-20000]
test_text = text[-20000:]

# print "Train size is: %s, testing size is: %s" % (len(train_text), len(test_text))
# mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=False, SINDEX=1, epoch=20)

# ce = mlp_ngram.crossentropy(test_text)
# print "Cross-entropy is:", ce
# print "Perplexity is:", numpy.exp(ce)

# print "Log rank is:", mlp_ngram.logaverank(test_text) 

s_prefix = u"拥挤"
top_tids, top_probs = mlp_ngram.topN(s_prefix, 10)
for x in top_tids:
	print nlpdict.gettoken(x),
print 
print top_probs