# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpNgram
@author: egg
'''

from nlpdict.NlpDict import NlpDict
from pylm.RnnLM import RnnLM
from pylm.RnnEmbLM import RnnEmbLM, RnnEmbTrLM
from pylm.MlpNgram import MlpNgram
from pylm.MlpBigram import MlpBigram
from pylm.Ngram import Ngram
import numpy
import time
import theano.sandbox.cuda


train_file_path = './data/pku_train_nw.ltxt'
print "Training file path:", train_file_path
# text
f = file(train_file_path)
text = unicode(f.read(), 'utf-8')
text = text.replace(" ", "")
f.close()


nlpdict = NlpDict()
nlpdict.buildfromtext(text, freq_thres=0)
print "NlpDict size is:", nlpdict.size()

theano.sandbox.cuda.use('gpu1')
# rnnlm = RnnLM(nlpdict, n_hidden=120, lr=0.05, batch_size=40, truncate_step=6, backup_file_path="./data/RnnLM/RnnLM.model.epoch71.n_hidden120.truncstep6.obj")
# mlp_bigram = MlpBigram(nlpdict, n_hidden=50, lr=0.13, batch_size=50, backup_file_path="./data/MlpBigram/MlpBigram.model.epoch500.n_hidden50.obj")
# mlp_ngram = MlpNgram(nlpdict, backup_file_path="./data/MlpNgram/Mlp4gram.model.epoch20.n_hidden150.obj", hvalue_file="./data/pku_embedding.obj")

mlp_ngram = MlpNgram(nlpdict, N=5, n_emb=100, n_hidden=400, lr=0.5, batch_size=200, dropout=False,
		backup_file_path='./data/MlpNgram/Mlp5gram.model.epoch100.n_hidden700.drFalse.n_emb100.in_size4702.obj')


# rnnlm = RnnEmbLM(nlpdict, 
# 		n_hidden=300, 
# 		lr=0.5, 
# 		batch_size=50, 
# 		truncate_step=4,
# 		dropout=True, 
# 		backup_file_path="./data/RnnEmbLM/RnnEmbLM.model.epoch17.n_hidden300.ss20.truncstep4.obj"
# 	)
# rnnlm.loadEmbeddings("./data/pku_embedding_rnn_c1.obj")

# rnnlm = RnnEmbTrLM(nlpdict, n_hidden=1200, lr=0.5, batch_size=100, truncate_step=4, 
# 		train_emb=True, dropout=True,
# 		backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch100.n_hidden1200.ssl20.truncstep4.drTrue.embsize100.obj"
# 	)


f = file("./test/Mlp5gram.model.epoch100.n_hidden700.drFalse.n_emb100.in_size4702.txt", 'wb')

def topN_predict(s_prefix):
	print >> f, "Test text:", s_prefix.encode("utf-8")
	top_tids, top_probs = mlp_ngram.topN(s_prefix, 10)
	for x in top_tids:
		print >> f, nlpdict.gettoken(x).encode("utf-8"),
	print >> f
	print >> f, top_probs
	print >> f

topN_predict(u"中共中央")
topN_predict(u"中国")
topN_predict(u"国际合作")
topN_predict(u"行政区")
topN_predict(u"建设工地")
topN_predict(u"装机")
topN_predict(u"搦管自思")
topN_predict(u"好囧")

topN_predict(u"中共中央总书记、国家主席")
topN_predict(u"一定能够取得改革开放的社会主义现代化建设的")
topN_predict(u"从城里来的人们听着惊心动魄的")
topN_predict(u"人们还在忍受战火的")
topN_predict(u"面临前所未有的大好囧")


f.close()