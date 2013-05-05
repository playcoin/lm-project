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

mlp_bigram = mlp_bigram = MlpBigram(nlpdict, n_hidden=30, lr=0.13, batch_size=50)

train_text = text
test_text = text[0:10]

print "MlpBigram train start!!"
for i in xrange(500):
	mlp_bigram.traintext(train_text, add_se=False, data_slice_size=1000)
	print "Error rate for test after epoch %s is %s" % (i+1, mlp_bigram.testtext(train_text))
	# print "Params: ", mlp_bigram.mlp.params[2].get_value()

print "MlpBigram train over!!"