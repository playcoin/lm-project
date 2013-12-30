# -*- coding: utf-8 -*-
'''
Created on 2013-11-28 13:33
@summary: 中文分词测试
@author: egg
'''

from nlpdict import NlpDict
from pyws import RnnWS
from pyws import RnnWFWS, RnnWFWS2, RnnWBWF2WS, RnnRevWS2, RnnFRWS
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
train_text = readClearFile("./data/datasets/msr_ws_train.ltxt")
nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

test_text = readClearFile("./data/datasets/msr_ws_test.ltxt")

fws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=600, lr=0.5, batch_size=150, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.0,# emb_dr_rate=0.1,
	backup_file_path="./data/model/RnnWFWS2.model.epoch10.n_hidden600.ssl20.truncstep4.dr0.0.embsize200.in_size5086.rc91.MSR.obj"
)

# rws = RnnRevWS2(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5,# emb_dr_rate=0.1,
# 	backup_file_path="./data/model/RnnRevWS2.model.epoch56.n_hidden1400.ssl20.truncstep4.dr0.5.embsize200.in_size4598.rc91.obj"
# )

result_file = "./data/result/5086_600_dr0_f.ltxt"

# frws = RnnFRWS(fws, rws)
#############
# Main Opr  #
#############
def main():

	sents = test_text.split('\n')
	print "Dict size is: %d, and sentences size is %d" % (nlpdict.size(), len(sents))
	# print rnnws.segment(sents[100], True)

	stime = time.clock()
	odtext = []
	for sent in sents:
		odtext.append(fws.segment(sent))

	writeFile(result_file, '\n'.join(odtext))

	print "Total time is %0.2fm." % ((time.clock() - stime) / 60.)

if __name__ == "__main__":
	main()