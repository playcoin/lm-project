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

theano.sandbox.cuda.use('gpu0')

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

rnnlm = RnnLM(nlpdict, n_hidden=30, lr=0.09, batch_size=50)

print "Rnn training start!"

train_text = text[:100]
test_text = text[:100]

s_time = time.clock()
for i in xrange(50):
	rnnlm.traintext(train_text)
	test_res = rnnlm.testtext(test_text)
	print "Error rate: %.5f. Epoch: %s. Training time so far: %0.1fm" % (test_res[0], i+1, (time.clock()-s_time)/60.)
	print ''.join(nlpdict.gettoken(i) for i in test_res[1])

e_time = time.clock()

duration = e_time - s_time

print "RnnLM train over!! The total training time is %.2fm." % (duration / 60.) 