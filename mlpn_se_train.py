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

print "Train size is: %s" % len_text

theano.sandbox.cuda.use('gpu1')

mlp_ngram = MlpNgram(nlpdict, N = 6, n_in = 30, n_hidden=70, lr=0.07, batch_size=50)

train_text = text
test_text = text[0:10000]

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True)

