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


train_file_path = './data/pku_train_nw.ltxt'
print "Training file path:", train_file_path
# text
f = file(train_file_path)
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

#############
# Trainging #
#############
train_text = text
test_text = text[:40000]

nlpdict = NlpDict()
nlpdict.buildfromtext(train_text, freq_thres=1)
print "NlpDict size is:", nlpdict.size()
print "Train size is: %s" % len(train_text)

rnnlm = RnnLM(nlpdict, n_hidden=150, lr=0.5, batch_size=50, truncate_step=4)
# rnnlm = RnnLM(nlpdict, n_hidden=200, lr=0.6, batch_size=40, truncate_step=6, backup_file_path="./data/RnnLM400000/RnnLM.model.epoch105.n_hidden200.truncstep6.obj")
# rnnlm = RnnLM(nlpdict, n_hidden=140, lr=0.05, batch_size=40, truncate_step=6, backup_file_path="./data/RnnLM/RnnLM.model.epoch49.n_hidden140.truncstep6.obj")
# rnnlm = RnnLM(nlpdict, n_hidden=120, lr=0.05, batch_size=40, truncate_step=6, backup_file_path="./data/RnnLM/RnnLM.model.epoch138.n_hidden150.truncstep6.obj")
# rnnlm = RnnLM(nlpdict, n_hidden=150, lr=0.13, batch_size=20, backup_file_path="./data/RnnLM/RnnLM.model.epoch138.n_hidden150.truncstep6.obj")
# rnnlm.lr = 0.0001
rnnlm.traintext(train_text, test_text, add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.92, DEBUG=True, SAVE=True, SINDEX=1)


#############
# rank test #
#############
# s_prefix = u"不堪"
# top_tids, top_probs= rnnlm.topN(s_prefix, 10)
# for x in top_tids:
# 	print nlpdict.gettoken(x),
# print 
# print top_probs
# print probs[-1][40]

#############
#  Testing  #
#############
# rnnlm = RnnLM(nlpdict, n_hidden=60, lr=0.13, batch_size=50, backup_file_path="./data/RnnLM/RnnLM.model.epoch21.n_hidden60.truncstep6.obj")
# simtest = u"""还在忍受战火"""
# print rnnlm.likelihood(simtest)
# # print rnnlm.ranks(simtest)
# # print rnnlm.logaverank(u"中共中央总书记、国家主席江泽民")
# pred, p_probs = rnnlm.predict(s_prefix)
# print pred, nlpdict.gettoken(pred)
# print p_probs[39:41]

f = file('./data/pku_test.txt')
tt = unicode(f.read(), 'utf-8')[]
f.close()

ce = rnnlm.crossentropy(tt)
print "Cross-entropy is:", ce
print "Perplexity is:", numpy.exp(ce)

print "Average log rank is:", rnnlm.logaverank(tt)
