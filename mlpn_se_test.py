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

nlpdict = NlpDict()
nlpdict.buildfromfile('./data/pku_train_nw.ltxt')


#############
# Trainging #
#############
# text
f = file('./data/pku_train_nw.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

len_text = len(text)
train_text = text

print "Train size is: %s" % len_text

theano.sandbox.cuda.use('gpu1')

mlp_ngram = MlpNgram(nlpdict, backup_file_path="./data/MlpNgram/Mlp4gram.model.epoch5.obj")
# mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=False)


#############
# Testing #
#############

f = file('./data/pku_test.txt')
test_text = unicode(f.read(), 'utf-8')
f.close()

ce = mlp_ngram.crossentropy(test_text[:100000])
print "Cross-entropy is:", ce
print "Perplexity is:", numpy.exp2(ce)