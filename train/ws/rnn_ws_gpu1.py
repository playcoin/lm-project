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
from fileutil import readClearFile, writeFile

import numpy
import time
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')

#############
# Data file #
#############
train_text = readClearFile("./data/datasets/pku_train_large_ws.ltxt")
train_tags = readClearFile("./data/datasets/pku_train_large_ws_tag.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

valid_text = readClearFile("./data/datasets/pku_valid_small_ws.ltxt")
valid_tags = readClearFile("./data/datasets/pku_valid_small_ws_tag.ltxt")

test_text = readClearFile("./data/datasets/pku_test_ws.ltxt")
test_tags = readClearFile("./data/datasets/pku_test_ws_tag.ltxt")

rnnws = RnnWFWS(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True, ext_emb=3,
	emb_file_path="./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4598.embeddings.obj"
)
rnnws.initRnn(dr_rate=0.3)
lr_coef = 0.91
r_init = "tremb.ext3.dr30.c91"


#############
# Main Opr  #
#############
def main():
	# 带验证集一起训练
	train_text = train_text + "\n" + valid_text
	train_tags = train_tags + "\n" + valid_tags

	print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))

	rnnws.traintext(train_text, train_tags, test_text, test_tags, 
		sen_slice_length=20, epoch=60, lr_coef=lr_coef, 
		DEBUG=True, SAVE=True, SINDEX=1, r_init=r_init
	)

if __name__ == "__main__":
	main()