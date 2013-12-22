# -*- coding: utf-8 -*-

from nlpdict import NlpDict

f = file('./data/datasets/pku_train_large_ws.ltxt')
train_text = unicode(f.read(), 'utf-8').replace(" ", "")

nlpdict = NlpDict(comb=True, combzh=True)
nlpdict.buildfromtext(train_text)	# 要先构造字典，把回车符给加进去
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))
f.close()


print nlpdict["0"]
print nlpdict[u"１"]

for i in range(10):
	print nlpdict.gettoken(i)

import cPickle

f = file('nddump', 'wb')
cPickle.dump(nlpdict.ndict_inv, f)
f.close()