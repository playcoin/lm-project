# -*- coding: utf-8 -*-


from nlpdict.NlpDict import NlpDict

nlpdict = NlpDict(comb=True, comben=False)
nlpdict.buildfromfile('./data/pku_train.ltxt', freq_thres=0)
print "Nlpdict size is:", nlpdict.size()

print nlpdict["0"]
print nlpdict[u"１"]
