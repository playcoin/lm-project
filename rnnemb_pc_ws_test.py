# -*- coding: utf-8 -*-
'''
Created on 2013-11-28 13:33
@summary: 中文分词测试
@author: egg
'''


from nlpdict.NlpDict import NlpDict
from pyws.RnnWS import RnnWS
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
train_text = train_text.replace("\n", "")	# 再把回车符给清空了
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))
f.close()

# tags
f = file('./data/pku_train_ws_tag.ltxt')
train_tags = unicode(f.read(), 'utf-8')
# 清空空格和回车
train_tags = train_tags.replace(" ", "").replace("\n", "")
f.close()

rnnws = RnnWS(nlpdict, n_emb=200, n_hidden=400, lr=0.15, batch_size=100, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=False,
	emb_file_path="./data/7gram.emb200.h1200.d4633.emb.obj"
)

rnnws.traintext(train_text, train_tags, train_text[:5000], train_tags[:5000], 
	sen_slice_length=20, epoch=50, lr_coef=0.925, 
	DEBUG=True, SAVE=False, SINDEX=1, r_init="7g200.c925"
)