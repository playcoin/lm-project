# -*- coding: utf-8 -*-
'''
Created on 2013-05-05 23:00
@summary: Test case on RnnLM
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.RnnEmbLM import RnnEmbTrLM
import numpy
import time


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
nlpdict.buildfromtext(text)
# nlpdict.transEmbedding('./data/pku_closed_word_embedding.ltxt', "./data/pku_embedding_s.obj")
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

rnnlm = RnnEmbTrLM(nlpdict, n_hidden=200, lr=0.5, batch_size=10, truncate_step=4, 
	emb_file_path="./data/pku_embedding_s.obj", dropout=False)
# rnnlm = RnnLM(nlpdict, n_hidden=200, lr=0.5, batch_size=10, truncate_step=4, dropout=True, backup_file_path="./data/simple_rnn_model.epoch150.n_hidden200.ts4.dylr.dropout.obj")

print "Rnn training start!"

rnnlm.traintext(train_text, test_text, add_se=False, sen_slice_length=10, epoch=150, lr_coef=0.96, DEBUG=True)
# rnnlm.savemodel("./data/simple_rnn_model.epoch150.n_hidden200.ts4.dylr.dropout.obj")