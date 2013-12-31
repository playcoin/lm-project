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

# No emb hidden_800 dr20
rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=800, lr=0.1, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.2,
	emb_file_path=None
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)

# No emb hidden_600 
rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=600, lr=0.1, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
	emb_file_path=None
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)

# No emb hidden_400 
rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=400, lr=0.1, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
	emb_file_path=None
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)

# No emb hidden_200 
rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=200, lr=0.1, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.0,
	emb_file_path=None
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)

# No emb hidden_1200 dr40
rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=1200, lr=0.1, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.4,
	emb_file_path=None
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)

# No emb hidden_1200 dr30
rnnlm = RnnEmbTrLM(nlpdict, n_emb=nlpdict.size(), n_hidden=1000, lr=0.1, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=False, dr_rate=0.3,
	emb_file_path=None
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)

# Train emb hidden_1000 dr40
rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1000, lr=0.5, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.4,
	emb_file_path="./data/7gram.emb200.h1200.d4633.emb.obj"
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)

# Train emb hidden_800 dr30
rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=800, lr=0.4, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.3,
	emb_file_path="./data/7gram.emb200.h1200.d4633.emb.obj"
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)

# Train emb hidden_600 dr20
rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=600, lr=0.3, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.2,
	emb_file_path="./data/7gram.emb200.h1200.d4633.emb.obj"
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)

# Train emb hidden_1400 dr60
rnnlm = RnnEmbTrLM(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=158, 
	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.55,
	emb_file_path="./data/7gram.emb200.h1200.d4633.emb.obj"
)
rnnlm.traintext(train_text, test_text, 
	add_se=False, sen_slice_length=20, epoch=100, lr_coef=0.94, 
	DEBUG=5, SAVE=5, SINDEX=1, r_init="c94"
)