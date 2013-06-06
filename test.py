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
ngram = Ngram(nlpdict, 3)
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

# f = file("./test/Ngram_test_output.txt", 'wb')

# def topN_predict(s_prefix):
# 	print >> f, "Test text:", s_prefix.encode("utf-8")
# 	top_tids, top_probs = ngram.topN(s_prefix, 10)
# 	for x in top_tids:
# 		print >> f, nlpdict.gettoken(x).encode("utf-8"),
# 	print >> f
# 	print >> f, top_probs
# 	print >> f

# topN_predict(u"中共中央")
# topN_predict(u"中国")
# topN_predict(u"国际合作")
# topN_predict(u"行政区")
# topN_predict(u"建设工地")
# topN_predict(u"装机")
# topN_predict(u"搦管自思")
# topN_predict(u"好囧")

# topN_predict(u"中共中央总书记、国家主席")
# topN_predict(u"一定能够取得改革开放的社会主义现代化建设的")
# topN_predict(u"从城里来的人们听着惊心动魄的")
# topN_predict(u"人们还在忍受战火的")


# f.close()


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
import random

sample_list = random.sample(range(len(test_text)), 1000)

rank_list = []
for i in sample_list:
	tokens = test_text[i:i+4]
	rank_list.append(ngram.ranks(tokens)[-1])

print numpy.log(rank_list).mean()