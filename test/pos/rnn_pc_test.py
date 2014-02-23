# -*- coding: utf-8 -*-
'''
Created on 2014-02-17 09:45
@summary: 词性标注测试
@author: Playcoin
'''


from nlpdict import NlpDict
from pypos import RnnPOS, RnnRevPOS
from fileutil import readClearFile, readFile, writeFile, dumpObj, loadObj
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

test_text = readClearFile("./data/datasets/pku_pos_test.ltxt")
test_tags = readFile("./data/datasets/pku_pos_test_tag.ltxt") # 不要清空格

rnnpos = RnnRevPOS(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=156, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5, emb_dr_rate=0.,
	backup_file_path="./data/model/RnnRevPOS.model.epoch60.n_hidden1400.ssl20.truncstep4.dr0.5.embsize200.in_size4598.rc90.obj"
)

#############
# Main Opr  #
#############
def main():
	lines = test_text.split('\n')
	print "Dict size is: %s. Test sentences is: %s" % (nlpdict.size(), len(lines))

	# rnnpos.acumPrior(train_tags)
	# olines = []

	# for line in lines:
	# 	olines.append(rnnpos.decode(line, True)) # for reverse

	# # writeFile('pypos/o1400_rev.ltxt', '\n'.join(olines))
	# dumpObj('pypos/1400_rev_c90_tpm.obj', olines)
	rnnpos.dumpembedding("data/RnnRevPOS.n_hidden1400.embsize200.in_size4598.embeddings.obj")


if __name__ == "__main__":
	main()