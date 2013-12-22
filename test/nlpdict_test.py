# -*- coding: utf-8 -*-

from nlpdict import NlpDict

nlpdict = NlpDict(comb=True, combzh=False)
nlpdict.buildfromfile('./data/datasets/pku_train.ltxt', freq_thres=0)
print "Nlpdict size is:", nlpdict.size()

print nlpdict["0"]
print nlpdict[u"１"]


import cPickle
f = file("./data/RnnWBWF2WS.model.epoch60.n_hidden1200.ssl20.truncstep4.drTrue.embsize200.in_size4598.rwbwf2.dr30.c91.obj")
data = cPickle.load(f)
f.close()