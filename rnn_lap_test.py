# -*- coding: utf-8 -*-
'''
Created on 2013-05-05 23:00
@summary: Test case on RnnLM
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.RnnLM import RnnLM
from pylm.RnnEmbLM import RnnEmbTrLM
import numpy
import time
import theano.sandbox.cuda


#############
# Trainging #
#############
# text
f = file('./data/text.txt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

train_text = text[:2405]
test_text = text[:2404]
len_text = len(train_text)

nlpdict = NlpDict()
nlpdict.buildfromtext(train_text)

print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

rnnlm = RnnLM(nlpdict, n_hidden=200, lr=0.5, batch_size=10, truncate_step=4, dropout=False)
# rnnlm = RnnLM(nlpdict, n_hidden=200, lr=0.5, batch_size=10, truncate_step=4, dropout=True, backup_file_path="./data/simple_rnn_model.epoch150.n_hidden200.ts4.dylr.dropout.obj")

rnnlm.traintext(train_text, test_text, add_se=False, sen_slice_length=20, epoch=50, lr_coef=0.96, DEBUG=True)
# print rnnlm.rnn.C.get_value().sum()
# rnnlm.savemodel("./data/simple_rnn_model.epoch150.n_hidden200.ts4.dylr.dropout.obj")

# print rnnlm.rnn.h_0.get_value()

# print rnnlm.testtext(test_text)[0]
s_prefix = u"中共中央"
print "Test text:", s_prefix
top_tids, top_probs = rnnlm.topN(s_prefix, 10)
for x in top_tids:
	print nlpdict.gettoken(x),
print
print top_probs
print