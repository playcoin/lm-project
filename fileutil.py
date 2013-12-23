# -*- coding: utf-8 -*-
'''
Created on 2013-12-22 18:40
@summary: File operations
@author: Playcoin
'''

def readClearFile(filepath):
	f = file(filepath, "rb")
	text = unicode(f.read(), "utf-8").replace(" ", "").replace("\r", "")
	f.close()
	return text

def readFile(filepath):
	f = file(filepath, "rb")
	text = unicode(f.read(), "utf-8").replace("\r", "")
	f.close()
	return text


def writeFile(filepath, content):
	# if is text, then should encode the text by utf-8
	f = file(filepath, "wb")
	f.write(content.encode("utf-8"))
	f.close()