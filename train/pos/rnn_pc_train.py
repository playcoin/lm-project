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
train_tags = readFile("./data/datasets/pku_pos_train_tag.ltxt")

rnnpos = RnnPOS(nlpdict, n_emb=200, n_hidden=400, lr=0.5, batch_size=5, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True, 
	emb_file_path="./data/RnnEmbTrLM.n_hidden1200.embsize200.in_size4598.embeddings.obj" 
)


#############
# Main Opr  #
#############
def main():
	print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))

	rnnpos.traintext(train_text[:1000], train_tags[:1000], train_text[:400], train_tags[:400], 
		sen_slice_length=20, epoch=60, lr_coef=0.92, 
		DEBUG=True, SAVE=False, SINDEX=1, r_init="7g200.c92"
	)

if __name__ == "__main__":
	main()