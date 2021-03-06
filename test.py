# -*- coding: utf-8 -*-
'''
Created on 2013-04-12 23:01
@summary: 测试用例
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.Ngram import Ngram
import numpy

nlpdict = NlpDict()
nlpdict.buildfromfile('./data/pku_train_nw.ltxt')

# text
f = file('./data/pku_train_nw.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

len_text = len(text)

print "Train size is: %s" % len_text

# ngram_file_path = "./data/ngram.model.obj"
ngram = Ngram(nlpdict, 2)
ngram.traintext(text)

# print "Save N-gram model"
# 
# ngram.savemodel(ngram_file_path)

# print nlpdict.gettoken(ngram.predict(u"国家主席")[0])
# ngram.traintokenseq(text)

# cal likelihood
prob_list = ngram.likelihood(u"国家主席江", False)
print "Likelihood of test sentence is:", prob_list
cet = - numpy.sum(numpy.log(prob_list)) / len(prob_list)
print "Cross-entropy of the test sentence is (interpolation):", cet

# cal likelihood
prob_list = ngram.likelihood(u"国家主席江", False, "backoff")
print "Likelihood of test sentence is:", prob_list
cet = - numpy.sum(numpy.log(prob_list)) / len(prob_list)
print "Cross-entropy of the test sentence is (backoff):", cet

# test text
f = file('./data/pku_test.txt')
test_text = unicode(f.read(), 'utf-8')
f.close()

print "Test size is: %s" % len(test_text)
ce, logs = ngram.crossentropy(test_text)
print "Cross-entropy is:", ce
print "Perplexity is:", numpy.exp2(ce)

s_prefix = u"国家"
prob = 0.
for char in nlpdict.ndict_inv:
	prob += ngram.likelihood(s_prefix+char, False, "interpolation")[-1]

print prob