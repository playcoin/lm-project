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
nlpdict.buildfromfile('./data/pku_train_nw.ltxt')

#############
# Trainging #
#############
# text
# f = file('./data/pku_train_nw.ltxt')
# text = unicode(f.read(), 'utf-8')
# text = text.replace(" ", "")
# f.close()

# len_text = len(text)

# print "Train size is: %s" % len_text

# rnnlm = RnnLM(nlpdict, n_hidden=40, lr=0.13, batch_size=40, truncate_step=6)
# rnnlm.traintext(train_text, test_text, add_se=False, epoch=50, DEBUG=True, SAVE=True)

#############
#  Testing  #
#############
rnnlm = RnnLM(nlpdict, n_hidden=40, lr=0.13, batch_size=40, backup_file_path="./data/RnnLM.model.epoch45.obj")
# print rnnlm.likelihood(u"中共中央总书记、国家主席江泽民")

print nlpdict.gettoken(rnnlm.predict(u"国家主席江"))

f = file('./data/pku_test.txt')
tt = unicode(f.read(), 'utf-8')
f.close()

ce = rnnlm.crossentropy(tt)
print "Cross-entropy is:", ce
print "Perplexity is:", numpy.exp2(ce)
