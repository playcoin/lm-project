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


#############
# Datafiles #
#############
# PKU small valid set
train_text = readClearFile("./data/datasets/pku_train_large_ws.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

test_text = readClearFile("./data/datasets/pku_test_ws.ltxt")

rnnws = RnnWBWF2WS(nlpdict, n_emb=200, n_hidden=1200, lr=0.5, batch_size=150, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dropout=True, #ext_emb=2,
	backup_file_path="./data/RnnWBWF2WS.model.epoch1.n_hidden1200.ssl20.truncstep4.drTrue.embsize200.in_size4598.rwbwf2.dr30.c91.new1.obj"
)
rnnws.initRnn(dr_rate=0.3)

result_file = "./data/results/decode_4598_e1.ltxt"


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

	writeFile(result_file, '\n'.join(odtext))

	print "Total time is %0.2fm." % ((time.clock() - stime) / 60.)

if __name__ == "__main__":
	main()