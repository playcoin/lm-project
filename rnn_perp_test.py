#-*- coding: utf-8 -*-
'''
Created on 2013-05-05 23:00
@summary: Test case on RnnLM
@author: Playcoin
'''

from nlpdict.NlpDict import NlpDict
from pylm.RnnLM import RnnLM
from pylm.RnnEmbLM import RnnEmbLM, RnnEmbTrLM
import numpy
import time
import theano.sandbox.cuda


train_file_path = './data/pku_train_nw.ltxt'
# train_file_path = './data/msr_training.ltxt'
print "Training file path:", train_file_path
# text
f = file(train_file_path)
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()

#############
# Trainging #
#############
train_text = text

nlpdict = NlpDict()
nlpdict.buildfromtext(train_text, freq_thres=0)
print "NlpDict size is:", nlpdict.size()
print "Train size is: %s" % len(train_text)

# rnnlm = RnnLM(nlpdict, 
# 		n_hidden=120, 
# 		lr=0.13, 
# 		batch_size=40, 
# 		truncate_step=4, 
# 		backup_file_path="./data/RnnLM/RnnLM.model.epoch71.n_hidden120.truncstep6.obj"
# 	)

# rnnlm = RnnEmbLM(nlpdict, 
# 		n_hidden=300, 
# 		lr=0.5, 
# 		batch_size=50, 
# 		truncate_step=4,
# 		dropout=True, 
# 		backup_file_path="./data/RnnEmbLM/RnnEmbLM.model.epoch29.n_hidden300.ssl20.truncstep4.drTrue.obj"
# 	)
# rnnlm.loadEmbeddings("./data/pku_embedding_rnn_c1.obj")

# theano.sandbox.cuda.use('gpu0')
# rnnlm = RnnEmbTrLM(nlpdict, n_hidden=500, lr=0.1, batch_size=120, truncate_step=4, 
# 		train_emb=True, dropout=False,
# 		backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch99.n_hidden500.ssl20.truncstep4.drFalse.embsize100.in_size4702.obj"
# 	)

# print rnnlm.testtext(train_text[:40000])

theano.sandbox.cuda.use('gpu1')
rnnlm = RnnEmbTrLM(nlpdict, n_hidden=1200, lr=0.6, batch_size=120, truncate_step=6, 
		train_emb=True, dropout=True,
		backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch100.n_hidden1200.ssl20.truncstep6.drTrue.embsize100.in_size4702.obj"
	)

# tt = u"天津市首届检察官艺术节音乐会日前举行。图为天津市检察官大合唱《检察官之歌》。（新华社记者李昌元摄）"

# probs = rnnlm.likelihood(tt, debug = True)

f = file('./data/pku_test.txt')
# f = file('./data/msr_test.ltxt')
tt = unicode(f.read(), 'utf-8')
f.close()

s_time = time.clock()
ce = rnnlm.crossentropy(tt)
ceo = ce * 174677 / 174305
# ceo = ce * 174677 / 174253
# ceo = ce * 188340 / 188293
ppl = numpy.exp(ce)
pplo = numpy.exp(ceo)
rankinfo = rnnlm.logaverank(tt)
e_time = time.clock()

print "PPL= %.6f, PPL(OOV)= %.6f, time cost for %s tokens is %.3fs" % (ppl, pplo, len(tt), (e_time - s_time))
print "Avelogrank= %.6f, rank1wSent= %.6f, rank5wSent= %.6f, rank10wSent= %.6f" % rankinfo