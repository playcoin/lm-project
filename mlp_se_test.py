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

nlpdict = NlpDict()
nlpdict.buildfromfile('./data/pku_train_nw.ltxt')

# text
f = file('./data/pku_train_nw.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

len_text = len(text)

print "Train size is: %s" % len_text

theano.sandbox.cuda.use('gpu')

mlp_bigram = MlpBigram(nlpdict, n_hidden=30, lr=0.07, batch_size=50)

train_text = text
test_text = text[0:10000]

print "MlpBigram train start!!"
s_time = time.clock()
for i in xrange(100):
	mlp_bigram.traintext(train_text, add_se=False)
	print "Error rate: %0.5f. Epoch: %s. Training time so far: %0.1fs" % (mlp_bigram.testtext(test_text), i+1, (time.clock()-s_time))
	# print "Params: ", mlp_bigram.mlp.params[2].get_value()
e_time = time.clock()

duration = e_time - s_time

print "MlpBigram train over!! The total training time is %.2fm." % (duration / 60.) 