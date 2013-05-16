# -*- coding: utf-8 -*-
'''
Created on 2013-04-28 15:53
@summary: test case for MlpBigram
@author: egg
'''

from nlpdict.NlpDict import NlpDict
from pylm.MlpBigram import MlpBigram
import numpy
import time
import theano.sandbox.cuda

nlpdict = NlpDict()
nlpdict.buildfromfile('./data/pku_train_nw.ltxt')


#############
# Trainging #
#############
# # text
# f = file('./data/pku_train_nw.ltxt')
# text = unicode(f.read(), 'utf-8')
# text = text.replace(" ", "")
# f.close()

# len_text = len(text)

# print "Train size is: %s" % len_text

# theano.sandbox.cuda.use('gpu0')

# mlp_bigram = MlpBigram(nlpdict, n_hidden=30, lr=0.13, batch_size=50)

# train_text = text
# test_text = text[0:10000]

# print "MlpBigram train start!!"
# s_time = time.clock()
# for i in xrange(50):
# 	mlp_bigram.traintext(train_text, add_se=False)
# 	print "Error rate: %0.5f. Epoch: %s. Training time so far: %0.1fm" % (mlp_bigram.testtext(test_text), i+1, (time.clock()-s_time)/60.)
# 	if (i+1) % 5 == 0:
# 		mlp_bigram.savemodel("./data/MlpBigram.model.epoch%s.obj" % i)

# e_time = time.clock()

# duration = e_time - s_time

# print "MlpBigram train over!! The total training time is %.2fm." % (duration / 60.) 

#############
#  Testing  #
#############

theano.sandbox.cuda.use('gpu0')
backup_file_path = "./data/MlpBigram.model.epoch44.obj"
print "Model file: ", backup_file_path
mlp_bigram = MlpBigram(nlpdict, backup_file_path=backup_file_path)
# print mlp_bigram.likelihood(u"国家主席江泽")

# print nlpdict.gettoken(mlp_bigram.predict(u"国家主席江"))

# test text
# f = file('./data/pku_test.txt')
# test_text = unicode(f.read(), 'utf-8')
# f.close()

# print "Test size is: %s" % len(test_text)
# ce, logs = mlp_bigram.crossentropy(test_text)
# print "Cross-entropy is:", ce
# print "Perplexity is:", numpy.exp2(ce)

mlp_bigram.savehvalues()