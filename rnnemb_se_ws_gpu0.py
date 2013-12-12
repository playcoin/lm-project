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
theano.sandbox.cuda.use('gpu0')

#############
# Trainging #
#############
# text
# f = file('./data/msr_training.ltxt')
f = file('./data/pku_train_ws.ltxt')
train_text = unicode(f.read(), 'utf-8')
# 清空空格和回车
train_text = train_text.replace(" ", "")
nlpdict = NlpDict(comb=True, comben=False)
nlpdict.buildfromtext(train_text)	# 要先构造字典，把回车符给加进去
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))
f.close()

# tags
f = file('./data/pku_train_ws_tag.ltxt')
train_tags = unicode(f.read(), 'utf-8')
# 清空空格和回车
train_tags = train_tags.replace(" ", "")
f.close()

#############
# Valid 	#
#############
f = file('./data/pku_valid_ws.ltxt')
valid_text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

f = file('./data/pku_valid_ws_tag.ltxt')
valid_tags = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()
#############
# Test  	#
#############
f = file('./data/pku_test_ws.ltxt')
test_text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

f = file('./data/pku_test_ws_tag.ltxt')
test_tags = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

# rnnws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True,
# 	backup_file_path="./data/RnnWFWS2/RnnWFWS2.model.epoch55.n_hidden1200.ssl20.truncstep4.drTrue.embsize200.in_size4633.r7g200.c93.obj"
# )

# sents = test_text.split('\n')
# otext = []
# odtext = []
# for sent in sents:
# 	otext.append(rnnws.segment(sent, False))
# 	odtext.append(rnnws.segment(sent, True))
# # tags
# f = file('./data/pku_test_output_ep55.ltxt', 'wb')
# f.write('\n'.join(otext).encode('utf-8'))
# f.close()

# # tags
# f = file('./data/pku_test_output_decode_ep55.ltxt', 'wb')
# f.write('\n'.join(odtext).encode('utf-8'))
# f.close()

# 先读RnnLM
rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1200, lr=0.2, batch_size=150, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True,
	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch100.n_hidden1200.ssl20.truncstep4.drTrue.embsize200.in_size4613.r7g200.c95.obj"
)
# 再读参数
# init_params = (rnnlm.rnnparams[0], rnnlm.rnnparams[1], None, rnnlm.rnnparams[3], None, None, None, None)
rnnlm.dumpembeddings("./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4613.embeddings.obj")

# 再初始化RnnWFWS2
rnnws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True,
	emb_file_path="./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4613.embeddings.obj"
)
# rnnws.rnnparams = init_params
# 带验证集一起训练
train_text = train_text + "\n" + valid_text
train_tags = train_tags + "\n" + valid_tags
print "Train size is: %s" % len(train_text)
rnnws.traintext(train_text, train_tags, test_text, test_tags, 
	sen_slice_length=20, epoch=60, lr_coef=0.92, 
	DEBUG=True, SAVE=True, SINDEX=1, r_init="nd4613.7g200.c92"
)