# -*- coding: utf-8 -*-
'''
Created on 2013-05-05 23:00
@summary: Test case on RnnLM
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.RnnLM import RnnLM
import numpy
import time
import theano.sandbox.cuda

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

print "Train size is: %s" % len_text

rnnlm = RnnLM(nlpdict, n_hidden=30, lr=0.1, batch_size=5)

print "Rnn training start!"

train_text = text[:201]

rnnlm.traintext(train_text, add_se=False, epoch=200, DEBUG=True)

# print rnnlm.testtext(test_text)[0]
