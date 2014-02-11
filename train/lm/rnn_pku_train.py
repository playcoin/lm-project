# -*- coding: utf-8 -*-
'''
Created on 2013-06-06 12:43
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
theano.sandbox.cuda.use('gpu0')

#############
# Trainging #
#############
# text
train_text = readClearFile("./data/datasets/pku_lm_train.ltxt")
nlpdict = NlpDict(comb=False, combzh=False, text=train_text)

valid_text = readClearFile("./data/datasets/pku_lm_valid.ltxt")

test_text = train_text[:5001]
len_text = len(train_text)

print "Dict size is: %s, Train size is: %s" % (nlpdict.size(), len_text)

# # No emb hidden_200 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=200, lr=0.5, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
# )

# # No emb hidden_250 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=250, lr=0.5, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
# )

# # No emb hidden_300 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=400, lr=0.5, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	backup_file_path="data/RnnEmbTrLM/RnnEmbTrLM.model.epoch85.n_hidden400.ssl20.truncstep4.dr0.0.embsize4633.in_size4633.rc96.obj"
# )
# rnnlm.lr = 0.5 * 0.96 ** 85
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=15, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=86, r_init="c96"
# )

# # No emb hidden_1200 dr40
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=1200, lr=0.4, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.4,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c96"
# )

# rnnlm = None
# gc.collect()

# # No emb hidden_350 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=350, lr=0.5, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c96"
# )

# rnnlm = None
# gc.collect()

# # No emb hidden_400 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=400, lr=0.5, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c96"
# )

# # No emb hidden_450 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=450, lr=0.3, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch100.n_hidden450.ssl20.truncstep4.dr0.0.embsize4633.in_size4633.rc962.obj"
# )
# rnnlm.lr = 0.3 * 0.962 ** 70
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=30, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=101, r_init="c94"
# )

# # No emb hidden_800 dr20
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=800, lr=0.1, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.2,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
# )

# # No emb hidden_600 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=600, lr=0.1, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
# )

# # No emb hidden_400 
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=400, lr=0.1, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
# )


# # No emb hidden_1200 dr30
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=1000, lr=0.1, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.3,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
# )


# # Train emb hidden_800 dr30
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=800, lr=0.4, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.3,
# 	emb_file_path="./data/7gram.emb200.h1200.d4633.emb.obj"
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
# )

# # Train emb hidden_600 dr20
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=600, lr=0.3, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.2,
# 	emb_file_path="./data/7gram.emb200.h1200.d4633.emb.obj"
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
# )

# # Train emb hidden_1000 dr40
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1000, lr=0.5, batch_size=158, 
# 	l2_reg=0.000000, truncate_step=4, train_emb=True, dr_rate=0.45,
# 	backup_file_path="data/RnnEmbTrLM/RnnEmbTrLM.model.epoch55.n_hidden1000.ssl20.truncstep4.dr0.45.embsize200.in_size4633.rc96.obj"
# )
# rnnlm.lr = 0.5 * 0.96 ** 55
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=45, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=56, r_init="c96"
# )

# Train emb hidden_1000 dr50
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1000, lr=0.5, batch_size=158, 
# 	l2_reg=0.000000, truncate_step=4, train_emb=True, dr_rate=0.5,
# 	backup_file_path="./data/RnnEmbTrLM/RnnEmbTrLM.model.epoch30.n_hidden1000.ssl20.truncstep4.dr0.5.embsize200.in_size4633.rc96.obj"
# )
# rnnlm.lr = 0.5 * 0.96 ** 30
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=70, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=31, r_init="c96"
# )

# # Train emb hidden_1400 dr60
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=158, 
# 	l2_reg=0.000000, truncate_step=4, train_emb=True, dr_rate=0.55,
# 	emb_file_path="./data/7gram.emb200.h1200.d4633.emb.obj"
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c96"
# )

##########################################
# for truncate_step                      #
##########################################
# # hidden_300 ts 6
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.5, batch_size=158, 
# 	l2_reg=0.000001, truncate_step=6, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
# )

# # hidden_300 ts 5
# rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=5, train_emb=False, dr_rate=0.0,
# 	emb_file_path=None
# )
# rnnlm.traintext(train_text, test_text, 
# 	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
# 	DEBUG=5, SAVE=5, SINDEX=1, r_init="c96"
# )

# hidden_300 ts 3
rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=300, lr=0.5, batch_size=158, 
	l2_reg=0.000001, truncate_step=3, train_emb=False, dr_rate=0.0,
	emb_file_path=None
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.96, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c96"
)
