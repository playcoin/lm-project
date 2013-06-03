# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpBigram
@author: egg
'''

from nlpdict.NlpDict import NlpDict
from pylm.MlpBigram import MlpBigram
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

# use gpu
theano.sandbox.cuda.use('gpu0')

# mlp_bigram = MlpBigram(nlpdict, n_hidden=50, lr=0.13, batch_size=50)
mlp_bigram = MlpBigram(nlpdict, n_hidden=50, lr=0.13, batch_size=50, backup_file_path="./data/MlpBigram/MlpBigram.model.epoch500.n_hidden50.obj")

train_text = text[:-20000]
test_text = text[-20000:]


# print "Train size is: %s, testing size is: %s" % (len(train_text), len(test_text))
# mlp_bigram.lr = 0.01
# mlp_bigram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=417, epoch=500)

# ce, logs = mlp_bigram.crossentropy(test_text)
# print "Cross-entropy is:", ce
# print "Perplexity is:", numpy.exp(ce)

#############
#  Testing  #
#############

# theano.sandbox.cuda.use('gpu0')
# backup_file_path = "./data/MlpBigram.model.epoch44.obj"
# print "Model file: ", backup_file_path
# mlp_bigram = MlpBigram(nlpdict, backup_file_path=backup_file_path)
# print mlp_bigram.likelihood(u"国家主席江泽")

# print nlpdict.gettoken(mlp_bigram.predict(u"国家主席江"))

# test text
# f = file('./data/pku_test.txt')
# test_text = unicode(f.read(), 'utf-8')
# f.close()

# print "Test size is: %s" % len(test_text)
# ce, logs = mlp_bigram.crossentropy(test_text)
# print "Cross-entropy is:", ce
# print "Perplexity is:", numpy.exp2(ce)

# mlp_bigram.savehvalues()

#############
# rank test #
#############
s_prefix = u"主"
top_tids, top_probs = mlp_bigram.topN(s_prefix, 20)
for x in top_tids:
	print nlpdict.gettoken(x),
print 
print top_probs