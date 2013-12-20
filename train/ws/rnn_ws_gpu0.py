# -*- coding: utf-8 -*-
'''
Created on 2013-11-28 13:33
@summary: 中文分词测试
@author: egg
'''


from nlpdict import NlpDict
from pyws import RnnWS
from pyws import RnnWFWS, RnnWFWS2, RnnWBWF2WS
from pylm import RnnEmbTrLM
import numpy
import time
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')

#############
# Trainging #
#############
# text
# f = file('./data/msr_training.ltxt')
f = file('./data/datasets/pku_train_large_ws.ltxt')
train_text = unicode(f.read(), 'utf-8')
# 清空空格和回车
train_text = train_text.replace(" ", "")
nlpdict = NlpDict(comb=True, combzh=True)
nlpdict.buildfromtext(train_text)	# 要先构造字典，把回车符给加进去
f.close()

# tags
f = file('./data/datasets/pku_train_large_ws_tag.ltxt')
train_tags = unicode(f.read(), 'utf-8')
# 清空空格和回车
train_tags = train_tags.replace(" ", "")
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

# rnnws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True,
# 	backup_file_path="./data/RnnWFWS2/RnnWFWS2.model.epoch55.n_hidden1200.ssl20.truncstep4.drTrue.embsize200.in_size4633.r7g200.c93.obj"
# )

# 带验证集一起训练
train_text = train_text + "\n" + valid_text
train_tags = train_tags + "\n" + valid_tags

print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True,
# 	emb_file_path="./data/7gram.emb200.h1200.d4598.emb.obj"
# )
# rnnlm.traintext(train_text, train_text[:501], 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
# 	DEBUG=True, SAVE=True, SINDEX=1, r_init="c94"
# )

# rnnlm.dumpembeddings("./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4598.embeddings.obj")

# # 再初始化RnnWFWS2
# rnnws = RnnWFWS(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True, ext_emb=2,
# 	emb_file_path="./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4598.embeddings.obj"
# )

# rnnws.initRnn(dr_rate=0.3)

# rnnws.traintext(train_text, train_tags, test_text, test_tags, 
# 	sen_slice_length=20, epoch=60, lr_coef=0.91, 
# 	DEBUG=True, SAVE=True, SINDEX=1, r_init="tremb.ext2.dr30.c91"
# )

rnnws = RnnWBWF2WS(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True,
	emb_file_path="./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4598.embeddings.obj"
)

rnnws.initRnn(dr_rate=0.3)

rnnws.traintext(train_text, train_tags, test_text, test_tags, 
	sen_slice_length=20, epoch=60, lr_coef=0.91, 
	DEBUG=True, SAVE=True, SINDEX=1, r_init="wbwf2.dr30.c91"
)