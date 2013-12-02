# -*- coding: utf-8 -*-
'''
Created on 2013-11-28 13:33
@summary: 中文分词测试
@author: egg
'''


from nlpdict.NlpDict import NlpDict
from pyws.RnnWS import RnnWS
from pylm.RnnEmbLM import RnnEmbTrLM
import numpy
import time
import theano.sandbox.cuda

#############
# Trainging #
#############
# text
# f = file('./data/msr_training.ltxt')
f = file('./data/pku_train_ws.ltxt')
train_text = unicode(f.read(), 'utf-8')
# 清空空格和回车
train_text = train_text.replace(" ", "")
nlpdict = NlpDict()
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
# rnnws = RnnWS(nlpdict, n_emb=50, n_hidden=200, lr=0.2, batch_size=50, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=False,
# 	backup_file_path="./data/RnnWS/RnnWS.model.epoch45.n_hidden200.ssl20.truncstep4.drFalse.embsize50.in_size4633.r7g50.c93.obj"
# )

f = file('./data/pku_valid_ws.ltxt')
valid_text = unicode(f.read(), 'utf-8')
# 清空空格和回车
valid_text = valid_text.replace(" ", "")
f.close()

# tags
f = file('./data/pku_valid_ws_tag.ltxt')
valid_tags = unicode(f.read(), 'utf-8')
# 清空空格和回车
valid_tags = valid_tags.replace(" ", "").replace("\n", "")
f.close()

# rnnws.testtext(valid_text[:1000], valid_tags[:1000])
rnnws = RnnWS(nlpdict, n_emb=50, n_hidden=300, lr=0.2, batch_size=50, 
	l2_reg=0.000001, truncate_step=4, train_emb=False, dropout=False,
	emb_file_path="./data/RnnEmbTrLM.n_hidden600.embsize50.in_size4633.emb.obj"
)

rnnws.traintext(train_text, train_tags, valid_text[:2000], valid_tags[:2000], 
	sen_slice_length=20, epoch=50, lr_coef=0.93, 
	DEBUG=True, SAVE=True, SINDEX=1, r_init="ntemb.7g50.c93"
)