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
train_text = readClearFile("./data/datasets/msr_ws_train.ltxt")
train_tags = readClearFile("./data/datasets/msr_ws_train_tag.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

valid_text = readClearFile("./data/datasets/msr_ws_valid.ltxt")
valid_tags = readClearFile("./data/datasets/msr_ws_valid_tag.ltxt")

test_text = readClearFile("./data/datasets/msr_ws_test.ltxt")
test_tags = readClearFile("./data/datasets/msr_ws_test_tag.ltxt")

rnnws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=600, lr=0.1, batch_size=150, 
	l2_reg=0.000000, truncate_step=4, train_emb=True, dr_rate=0.0,
	emb_file_path="./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size5086.embeddings.obj"
)
lr_coef = 0.94
r_init = "c94.MSR"


#############
# Main Opr  #
#############
def main():
	# 带验证集一起训练
	global train_text, train_tags
	train_text = train_text# + "\n" + valid_text
	train_tags = train_tags# + "\n" + valid_tags

	print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))

	rnnws.traintext(train_text, train_tags, train_text[:5000], train_tags[:5000], 
		sen_slice_length=20, epoch=30, lr_coef=lr_coef, 
		DEBUG=True, SAVE=True, SINDEX=1, r_init=r_init
	)

if __name__ == "__main__":
	main()