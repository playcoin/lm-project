# -*- coding: utf-8 -*-
'''
Created on 2013-11-28 13:33
@summary: 中文分词测试
@author: egg
'''

from nlpdict import NlpDict
from pyws import RnnWS
from pyws import RnnWFWS, RnnWFWS2, RnnWBWF2WS, RnnRevWS2
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

#############
# Main Opr  #
#############
def main():
	# 带验证集一起训练
	global train_text, train_tags, test_text, test_tags
	train_text = train_text + "\n" + valid_text
	train_tags = train_tags + "\n" + valid_tags

	print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))

	rnnws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=600, lr=0.3, batch_size=158, 
		l2_reg=0.000000, truncate_step=4, train_emb=True, dr_rate=0.0,
		emb_file_path=None
	)

	rnnws.traintext(train_text, train_tags, test_text[:1000], test_tags[:1000], 
		sen_slice_length=20, epoch=70, lr_coef=0.91, 
		DEBUG=5, SAVE=5, SINDEX=1, r_init="c91"
	)

	# rnnws = RnnWFWS2(nlpdict, n_emb=100, n_hidden=300, lr=0.5, batch_size=158, 
	# 	l2_reg=0.000000, truncate_step=4, train_emb=True, dr_rate=0.0,
	# 	emb_file_path=None
	# )

	# rnnws.traintext(train_text, train_tags, test_text[:1000], test_tags[:1000], 
	# 	sen_slice_length=20, epoch=70, lr_coef=0.91, 
	# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c91"
	# )

if __name__ == "__main__":
	main()