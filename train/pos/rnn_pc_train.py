# -*- coding: utf-8 -*-
'''
Created on 2013-12-23 13:49
@summary: 中文词性标注测试
@author: Playcoin
'''

from nlpdict import NlpDict
from pypos import RnnPOS
from fileutil import readClearFile, readFile, writeFile
import numpy
import time
import theano.sandbox.cuda

#############
# Datafiles #
#############
nlpdict_text = readClearFile("./data/datasets/pku_ws_train_large.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=nlpdict_text)

train_text = readClearFile("./data/datasets/pku_pos_train.ltxt")
train_tags = readFile("./data/datasets/pku_pos_train_tag.ltxt") # 不要清空格

rnnpos = RnnPOS(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=10, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True, #ext_emb=2,
	backup_file_path="./data/model/RnnWFWS2.model.epoch60.n_hidden1400.ssl20.truncstep4.drTrue.embsize200.in_size4598.rtremb.c91.obj"
)
rnnpos.batch_size = 10
rnnpos.rnnparams[2] = None # W_out
rnnpos.rnnparams[4] = None # b_out
rnnpos.rnnparams[5] = None # h_0, batch_size变了

#############
# Main Opr  #
#############
def main():
	print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))

	rnnpos.traintext(train_text[:2000], train_tags[:20000], train_text[:800], train_tags[:8000], 
		sen_slice_length=20, epoch=60, lr_coef=0.92, 
		DEBUG=True, SAVE=False, SINDEX=1, r_init="7g200.c92"
	)

if __name__ == "__main__":
	main()