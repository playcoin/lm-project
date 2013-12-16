# -*- coding: utf-8 -*-
'''
Created on 2013-11-28 13:33
@summary: 中文分词测试
@author: egg
'''


from nlpdict.NlpDict import NlpDict
from pyws.RnnWS import RnnWS
from pyws.RnnWFWS import RnnWFWS, RnnWFWS2
from pylm.RnnEmbLM import RnnEmbTrLM
import numpy
import time
import theano.sandbox.cuda

#############
# Trainging #
#############
# text
# f = file('./data/msr_training.ltxt')
f = file('./data/datasets/pku_train_large_ws.ltxt')
train_text = unicode(f.read(), 'utf-8').replace(" ", "")

nlpdict = NlpDict(comb=True, comben=True)
nlpdict.buildfromtext(train_text)	# 要先构造字典，把回车符给加进去
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))
f.close()

# tags
f = file('./data/datasets/pku_train_large_ws_tag.ltxt')
train_tags = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

#############
# Valid 	#
#############
# rnnws = RnnWS(nlpdict, n_emb=50, n_hidden=200, lr=0.2, batch_size=50, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=False,
# 	backup_file_path="./data/RnnWS/RnnWS.model.epoch45.n_hidden200.ssl20.truncstep4.drFalse.embsize50.in_size4633.r7g50.c93.obj"
# )

f = file('./data/datasets/pku_test_ws.ltxt')
test_text = unicode(f.read(), 'utf-8')
# 清空空格和回车
test_text = test_text.replace(" ", "")
f.close()

# tags
f = file('./data/datasets/pku_test_ws_tag.ltxt')
test_tags = unicode(f.read(), 'utf-8')
# 清空空格和回车
test_tags = test_tags.replace(" ", "")
f.close()

rnnws = RnnWFWS(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=150, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True, ext_emb=2,
	backup_file_path="./data/models/RnnWFWS2.model.epoch42.n_hidden1200.ssl20.truncstep4.drTrue.embsize200.in_size4598.rtremb.c91.obj"
)

sents = test_text.split('\n')[:100]

print rnnws.segment(sents[0], True)

stime = time.clock()
otext = []
odtext = []
for sent in sents:
	# otext.append(rnnws.segment(sent, False))
	odtext.append(rnnws.segment(sent, True))
# tags
# f = file('./data/results/pku_test_output_4598_e15.ltxt', 'wb')
# f.write('\n'.join(otext).encode('utf-8'))
# f.close()

# tags
f = file('./data/results/pku_test_output_decode_test_1.ltxt', 'wb')
f.write('\n'.join(odtext).encode('utf-8'))
f.close()

print "Total time is %0.2fm." % ((time.clock() - stime) / 60.)