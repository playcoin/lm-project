# -*- coding: utf-8 -*-
'''
Created on 2013-12-02 15:35
@summary: FNN 中文测试
@author: egg
'''

from nlpdict import NlpDict
from pyws import MlpWS
import numpy
import time
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')
#############
# Trainging #
#############
# text
f = file('data/pku_train_ws.ltxt')
train_text = unicode(f.read(), 'utf-8')
train_text = train_text.replace(" ", "")
nlpdict = NlpDict()
nlpdict.buildfromtext(train_text)	# 要先构造字典，把回车符给加进去
print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))
f.close()

# tags
f = file('data/pku_train_ws_tag.ltxt')
train_tags = unicode(f.read(), 'utf-8')
# 清空空格和回车
train_tags = train_tags.replace(" ", "")
f.close()

f = file('./data/pku_test_ws.ltxt')
test_text = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

# tags
f = file('./data/pku_test_ws_tag.ltxt')
test_tags = unicode(f.read(), 'utf-8').replace(" ", "")
f.close()

mlp_ws = MlpWS(nlpdict, chunk_size=5, n_emb=50, n_hidden=200, lr=0.2, batch_size=50, dropout=False,
		emb_file_path='data/7gram.emb50.h900.d4633.emb.obj')

mlp_ws.traintext(train_text, train_tags, test_text, test_tags, DEBUG=True, SAVE=False, SINDEX=1, epoch=100, lr_coef=0.94)
