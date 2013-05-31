# -*- coding: utf-8 -*-


from nlpdict.NlpDict import NlpDict

nlpdict = NlpDict()
nlpdict.buildfromfile('./data/pku_train_nw.ltxt',freq_thres=0)
print "Nlpdict size is:", nlpdict.size()

nlpdict.transEmbedding("./data/pku_closed_word_embedding.ltxt", "./data/pku_embedding.obj")