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
import re
import theano.sandbox.cuda


#############
# Datafiles #
#############
# PKU small valid set
train_text = readClearFile("./data/datasets/pku_ws_train_large.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

test_text = readClearFile("./data/datasets/pku_ws_test.ltxt")

rnnws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=150, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True, #ext_emb=2,
	backup_file_path="./data/model/RnnWFWS2.model.epoch60.n_hidden1400.ssl20.truncstep4.drTrue.embsize200.in_size4598.rtremb.c91.obj"
)
rnnws.initRnn(dr_rate=0.5)

result_file = "./data/result/decode_4598_1400_dr50_ext2.ltxt"


#############
# Main Opr  #
#############
def main():

	sents = test_text.split('\n')
	print "Dict size is: %d, and sentences size is %d" % (nlpdict.size(), len(sents))

	stime = time.clock()
	odtext = []
	for sent in sents:
		odtext.append(rnnws.segment(sent, True))

	# text = re.sub(r"(\d)  \.  (\d)", r"\1.\2", '\n'.join(odtext))

	writeFile(result_file, '\n'.join(odtext))

	print "Total time is %0.2fm." % ((time.clock() - stime) / 60.)

if __name__ == "__main__":
	main()