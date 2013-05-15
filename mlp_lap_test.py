# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpBigram and MlpNgram
@author: egg
'''

from nlpdict.NlpDict import NlpDict
from pylm.MlpNgram import MlpNgram
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

mlp_ngram = MlpNgram(nlpdict, N = 4)

mat_in, label = mlp_ngram.tids2inputdata(range(18, 41), zero_start=False, truncate_input=True)

print mat_in.get_value(), mat_in.get_value().shape
print label.get_value(), label.get_value().shape