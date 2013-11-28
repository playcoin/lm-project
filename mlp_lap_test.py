# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpBigram and MlpNgram
@author: egg
'''

from nlpdict.NlpDict import NlpDict
from pylm.MlpBigram import MlpBigram
import numpy
import theano.sandbox.cuda

# theano.sandbox.cuda.use('gpu')

nlpdict = NlpDict()
nlpdict.buildfromfile('./data/text.txt')

#############
# Trainging #
#############
# text
f = file('./data/text.txt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

len_text = len(text)
train_text = text[:2405]
test_text = text[2405:]

print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

mlp_bigram = MlpBigram(nlpdict, n_hidden=40, lr=.5)

mlp_bigram.traintext(train_text, test_text, DEBUG=True, SAVE=False, SINDEX=1, epoch=500)