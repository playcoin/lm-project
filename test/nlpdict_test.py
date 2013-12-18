# -*- coding: utf-8 -*-

from nlpdict import NlpDict

nlpdict = NlpDict(comb=True, combzh=False)
nlpdict.buildfromfile('./data/datasets/pku_train.ltxt', freq_thres=0)
print "Nlpdict size is:", nlpdict.size()

print nlpdict["0"]
print nlpdict[u"１"]
