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

rnnws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=128, 
	l2_reg=0.000000, truncate_step=4, train_emb=True, dr_rate=0.3,
	backup_file_path="./data/RnnWFWS2/RnnWFWS2.model.epoch40.n_hidden1200.ssl20.truncstep4.dr0.3.embsize200.in_size5086.rc91.MSR.obj"
)
rnnws.lr = 0.5 * 0.91 ** 45
lr_coef = 0.91
r_init = "c91.MSR"


#############
# Main Opr  #
#############
def main():
	# 带验证集一起训练
	global train_text, train_tags, test_text, test_tags
	train_text = train_text + "\n" + valid_text
	train_tags = train_tags + "\n" + valid_tags
	print len(train_text), len(test_text)

	print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))

	rnnws.traintext(train_text, train_tags, test_text, test_tags, 
		sen_slice_length=20, epoch=5, lr_coef=lr_coef, 
		DEBUG=2, SAVE=5, SINDEX=46, r_init=r_init
	)

if __name__ == "__main__":
	main()