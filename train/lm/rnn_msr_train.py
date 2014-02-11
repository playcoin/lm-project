# -*- coding: utf-8 -*-
'''
Created on 2013-06-13 21:21
@summary: 
@author: Playcoin
'''

from nlpdict import NlpDict
from pylm import RnnEmbTrLM
from pylm import MlpNgram
from fileutil import readClearFile

import gc
import numpy
import time
import theano.sandbox.cuda
# use gpu
theano.sandbox.cuda.use('gpu1')

#############
# Trainging #
#############
# text
train_text = readClearFile("./data/datasets/msr_lm_train.ltxt")
nlpdict = NlpDict(comb=False, combzh=False, text=train_text)

test_text = train_text[:5001]
print "Dict size is: %s, Training size is: %s" % (nlpdict.size(), len(train_text))


# # No emb hidden_200
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=200, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=50, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94.MSR"
# )


# # No emb hidden_300 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.4, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	backup_file_path="data/RnnEmbTrLM/RnnEmbTrLM.model.epoch50.n_hidden300.ssl20.truncstep4.dr0.0.embsize5127.in_size5127.rc94.MSR.obj"
# )
# rnnlm.lr = 0.4 * 0.94 ** 40
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=10, lr_coef=0.94, 
# 	DEBUG=1, SAVE=5, SINDEX=51, r_init="c94.MSR"
# )

# # No emb hidden_350 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=350, lr=0.4, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	backup_file_path="data/RnnEmbTrLM/RnnEmbTrLM.model.epoch45.n_hidden350.ssl20.truncstep4.dr0.0.embsize5127.in_size5127.rc942.MSR.obj"
# )
# rnnlm.lr = 0.4 * 0.942 ** 45
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=15, lr_coef=0.942, 
# 	DEBUG=5, SAVE=5, SINDEX=46, r_init="c942.MSR"
# )

# # No emb hidden_400 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=400, lr=0.38, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=60, lr_coef=0.945, 
# 	DEBUG=1, SAVE=5, SINDEX=1, r_init="c942.MSR"
# )
# No emb hidden_450 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=450, lr=0.20, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	backup_file_path="data/RnnEmbTrLM/RnnEmbTrLM.model.epoch60.n_hidden450.ssl20.truncstep4.dr0.0.embsize5127.in_size5127.rc947.MSR.obj"
# )
# rnnlm.lr = 0.5 * 0.94 ** 20
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=40, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=61, r_init="c94.MSR"
# )

# # No emb hidden_250 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=250, lr=0.4, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch30.n_hidden250.ssl20.truncstep4.dr0.0.embsize5127.in_size5127.rc94.MSR.obj"
# )
# rnnlm.lr = 0.5 * 0.94 ** 30
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=30, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=31, r_init="c94.MSR"
# )


# # No emb hidden_1200 dr40
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=1200, lr=0.1, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.4,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=50, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94.MSR"
# )

# # No emb hidden_1200 dr30
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=1000, lr=0.1, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.3,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=50, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94.MSR"
# )

# # Train emb hidden_1000 dr40
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1000, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.4,
# 	emb_file_path="./data/7gram.emb200.h1200.d5172.emb.obj"
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=50, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94.MSR"
# )

# # Train emb hidden_800 dr30
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=800, lr=0.4, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.3,
# 	emb_file_path="./data/7gram.emb200.h1200.d5172.emb.obj"
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=50, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94.MSR"
# )

# # Train emb hidden_600 dr20
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=600, lr=0.3, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.2,
# 	emb_file_path="./data/7gram.emb200.h1200.d5172.emb.obj"
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=50, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94.MSR"
# )

# # Train emb hidden_1400 dr55
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.55,
# 	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch30.n_hidden1400.ssl20.truncstep4.dr0.55.embsize200.in_size5127.rc94.MSR.obj"
# )
# rnnlm.lr = 0.5 * 0.94 ** 30
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=30, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=31, r_init="c94.MSR"
# )

# # ts6 No emb hidden_300 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.4, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=6, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=60, lr_coef=0.94, 
# 	DEBUG=1, SAVE=5, SINDEX=1, r_init="c94.MSR"
# )

# rnnlm = None

# ts5 No emb hidden_300 
rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.4, batch_size=150, 
	l2_reg=0.000001, truncate_step=5, train_emb=False, dr_rate=0.0,
	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch45.n_hidden300.ssl20.truncstep5.dr0.0.embsize5127.in_size5127.rc94.MSR.obj"
)
rnnlm.lr = 0.4 * 0.94 ** 45
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=15, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=46, r_init="c94.MSR"
)

rnnlm = None
# ts3 No emb hidden_300 
rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.4, batch_size=150, 
	l2_reg=0.000001, truncate_step=3, train_emb=False, dr_rate=0.0,
	emb_file_path=None
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=60, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94.MSR"
)

