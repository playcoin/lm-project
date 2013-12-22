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

def writeFile(filepath, content):
	# if is text, then should encode the text by utf-8
	content = type(content) == str and content.encode("utf-8") or content
	
	f = file(filepath, "wb")
	f.write(content)
	f.close()