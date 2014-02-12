# -*- coding: utf-8 -*-
'''
Created on 2013-12-23 13:39
@summary: 词性标注器
@author: Playcoin
'''

import numpy
import cPickle
import time
import re

import theano
from pypos import RnnPOS
from netagutil import tagmap, tagsize

class RnnNER(RnnPOS):
	'''
	@summary: 词性标注器
	'''

	def __init__(self, *args, **kwargs):
		super(RnnNER, self).__init__(*args, **kwargs)

		self.out_size = tagsize
		self.septag = tagmap["o"]
		self.tagsize = tagsize