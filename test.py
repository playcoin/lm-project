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
nlpdict.buildfromfile('./data/pku_train_nw.ltxt',freq_thres=0)
print "Nlpdict size is:", nlpdict.size()

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
# prob_list = ngram.likelihood(u"国家主席江", False)
# print "Likelihood of test sentence is:", prob_list
# cet = - numpy.sum(numpy.log(prob_list)) / len(prob_list)
# print "Cross-entropy of the test sentence is (interpolation):", cet

# # cal likelihood
# prob_list = ngram.likelihood(u"国家主席江", False, "backoff")
# print "Likelihood of test sentence is:", prob_list
# cet = - numpy.sum(numpy.log(prob_list)) / len(prob_list)
# print "Cross-entropy of the test sentence is (backoff):", cet

# test text
f = file('./data/pku_test.txt')
test_text = unicode(f.read(), 'utf-8')
f.close()

# print "Test size is: %s" % len(test_text)
# ce, logs = ngram.crossentropy(test_text)
# print "Cross-entropy is:", ce
# print "Perplexity is:", numpy.exp(ce)

# for i in xrange(0, len(test_text), 10000):
# 	ce, logs = ngram.crossentropy(test_text[i:i+10000])
# 	print "Cross-entropy of %s~%s is %s:" % (i, i+10000, ce) 

# # ce, logs = ngram.crossentropy(test_text[87730:87740])
# # print "Cross-entropy of %s~%s is %s:" % (87730, 87740, ce) 



# s_prefix = u"囧家"
# prob = []
# for char in nlpdict.ndict_inv:
# 	prob.append(ngram.likelihood(s_prefix+char, False)[-1])

# print prob

# s_prefix = u"的戊"
# prob = 0.
# for char in nlpdict.ndict_inv:
# 	prob += ngram.likelihood(s_prefix+char, False)[-1]

# print prob

#############
# rank test #
#############
s_prefix = u"主席"
tids = [nlpdict[x] for x in s_prefix]
top_tids, top_probs = ngram.topN(tids)
for x in top_tids:
	print nlpdict.gettoken(x),
print 
print top_probs
# rank_list = []
# for i in xrange(nlpdict.size()):
# 	rank_list.append(ngram.rank(tids + [i]))

# print rank_list


# tokenids = [nlpdict[x] for x in u"国家主"]

# print ngram.rank([20,21,22])

# print ngram.rank([20,21,22])

# rank_list = ngram.ranks(test_text[:10000])

# print numpy.log(rank_list).mean()


###################
# PPL by sampling #
###################
# import random

# sample_list = random.sample(range(len(test_text)), 1000)

# rank_list = []
# for i in sample_list:
# 	tokens = test_text[i:i+4]
# 	rank_list.append(ngram.ranks(tokens)[-1])

# print numpy.log(rank_list).mean()