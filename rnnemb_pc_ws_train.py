# -*- coding: utf-8 -*-
'''
Created on 2013-12-12 13:33
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
f = file('./data/datasets/pku_train_large_ws.ltxt')
train_text = unicode(f.read(), 'utf-8').replace(" ", "")
# 要先构造字典，把回车符给加进去
nlpdict = NlpDict(comb=True)
nlpdict.buildfromtext(train_text)	
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))
f.close()

# tags
f = file('./data/datasets/pku_train_large_ws_tag.ltxt')
train_tags = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

#############
# Valid 	#
#############
f = file('./data/datasets/pku_valid_small_ws.ltxt')
valid_text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

f = file('./data/datasets/pku_valid_small_ws_tag.ltxt')
valid_tags = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()
#############
# Test  	#
#############
f = file('./data/datasets/pku_test_ws.ltxt')
test_text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

f = file('./data/datasets/pku_test_ws_tag.ltxt')
test_tags = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

# 再初始化RnnWFWS2
rnnws = RnnWFWS(nlpdict, n_emb=200, n_hidden=300, lr=0.5, batch_size=10, 
	l2_reg=0.000001, truncate_step=4, train_emb=False, dropout=False, ext_emb=2,
	emb_file_path="./data/embeddings/RnnEmbTrLM.n_hidden1200.embsize200.in_size4598.embeddings.obj"
)

# rnnws.rnnparams = init_params
# 带验证集一起训练
# train_text = train_text + "\n" + valid_text
# train_tags = train_tags + "\n" + valid_tags
train_text = train_text[:2500]
train_tags = train_tags[:2500]
print "Train size is: %s" % len(train_text)
rnnws.traintext(train_text, train_tags, train_text[:1000], train_tags[:1000], 
	sen_slice_length=20, epoch=60, lr_coef=0.92, 
	DEBUG=True, SAVE=False, SINDEX=1, r_init="nd4613.7g200.c92"
)