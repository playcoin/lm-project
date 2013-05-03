# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpBigram
@author: egg
'''

from nlpdict.NlpDict import NlpDict
from pylm.MlpBigram import MlpBigram
import numpy

nlpdict = NlpDict()
nlpdict.buildfromfile('./data/pku_train_s.ltxt')

# text
f = file('./data/pku_train_s.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

len_text = len(text)

print "Train size is: %s" % len_text

mlp_bigram = MlpBigram(nlpdict, n_hidden=10, lr=0.05)

train_text = text[:]
test_text = text[0:10]

print "MlpBigram train start!!"
for i in xrange(500):
	mlp_bigram.traintext(train_text, batch_size=10, add_se=False, gpu=True)
	print "Error rate for test after epoch %s is %s" % (i+1, mlp_bigram.testtext(train_text)[0])
	# print "Params: ", mlp_bigram.mlp.params[2].get_value()

print "MlpBigram train over!!"