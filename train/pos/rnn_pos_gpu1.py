# -*- coding: utf-8 -*-
'''
Created on 2013-12-26 09:45
@summary: 词性标注测试
@author: Playcoin
'''


from nlpdict import NlpDict
from pypos import RnnPOS, RnnRevPOS
from fileutil import readClearFile, readFile, writeFile
import numpy
import time
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')

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
	emb_file_path="./data/RnnWFWS2.n_hidden1400.embsize200.in_size4598.embeddings.obj"
)


#############
# Main Opr  #
#############
def main():
	print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len(train_text))
	tlines= test_tags.split('\n')[:500]
	lines = test_text.split('\n')[:500]

	rnnpos.traintext(train_text, train_tags, '\n'.join(lines), '\n'.join(tlines), 
		sen_slice_length=20, epoch=60, lr_coef=0.92, 
		DEBUG=1, SAVE=5, SINDEX=1, r_init="c92"
	)
	# for (line, tline) in zip(lines, tlines):
	# 	err = rnnpos.testtext(line, tline)
	# 	if err > 0.9:
	# 		print line
	# 		print tline

	# rnnpos.testtext('\n'.join(lines), '\n'.join(tlines))



if __name__ == "__main__":
	main()