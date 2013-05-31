# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpNgram
@author: egg
'''

from nlpdict.NlpDict import NlpDict
from pylm.MlpNgram import MlpNgram
import numpy
import time
import theano.sandbox.cuda

nlpdict = NlpDict()
nlpdict.buildfromfile('./data/pku_train_nw.ltxt')


#############
# Trainging #
#############
# text
f = file('./data/pku_train_nw.ltxt')
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

len_text = len(text)

print "Train size is: %s" % len_text

theano.sandbox.cuda.use('gpu1')

mlp_ngram = MlpNgram(nlpdict, N=5, n_in=50, n_hidden=100, lr=0.07, batch_size=50, hvalue_file="./data/pku_embedding.obj")
# mlp_ngram = MlpNgram(nlpdict, backup_file_path="./data/MlpNgram/Mlp5gram.model.epoch10.n_hidden100.obj", hvalue_file="./data/pku_embedding.obj")
mlp_ngram.lr = 0.001

train_text = text
test_text = text[0:20000]

mlp_ngram.traintext(train_text, test_text, DEBUG=True, SAVE=True, SINDEX=11)

