# -*- coding: utf-8 -*-
'''
Created on 2013-04-12 23:01
@summary: 测试用例
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.Ngram import Ngram

nlpdict = NlpDict()
nlpdict.buildfromfile('./data/pku_train.ltxt')

# text
f = file('./data/pku_train.ltxt')
text = unicode(f.read(), 'utf-8')
text.replace(" ", "")
f.close()

ngram = Ngram(nlpdict, 4)
ngram.traintext(text)

