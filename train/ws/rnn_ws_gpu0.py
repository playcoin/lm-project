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
theano.sandbox.cuda.use('gpu0')

#############
# Data file #
#############
train_text = readClearFile("./data/datasets/pku_ws_train_large.ltxt")
train_tags = readClearFile("./data/datasets/pku_ws_train_large_tag.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

valid_text = readClearFile("./data/datasets/pku_ws_valid_small.ltxt")
valid_tags = readClearFile("./data/datasets/pku_ws_valid_small_tag.ltxt")

test_text = readClearFile("./data/datasets/pku_ws_test.ltxt")
test_tags = readClearFile("./data/datasets/pku_ws_test_tag.ltxt")

rnnws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5,
	emb_file_path="./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4598.embeddings.obj"
)
lr_coef = 0.91
r_init = "c91"


#############
# Main Opr  #
#############
def main():
	# 带验证集一起训练
	global train_text, train_tags
	train_text = train_text + "\n" + valid_text
	train_tags = train_tags + "\n" + valid_tags

	print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))

	rnnws.traintext(train_text, train_tags, test_text, test_tags, 
		sen_slice_length=20, epoch=60, lr_coef=lr_coef, 
		DEBUG=True, SAVE=True, SINDEX=1, r_init=r_init
	)

	###############
	# test emb_dr #
	###############
	rnnws1 = RnnWFWS2(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=158, 
		l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5, emb_dr_rate=0.1,
		emb_file_path="./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4598.embeddings.obj"
	)
	rnnws1.traintext(train_text, train_tags, test_text, test_tags, 
		sen_slice_length=20, epoch=60, lr_coef=0.91, 
		DEBUG=True, SAVE=True, SINDEX=1, r_init="embdr0.1.c91"
	)

if __name__ == "__main__":
	main()