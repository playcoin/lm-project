# -*- coding: utf-8 -*-
'''
Created on 2013-11-28 11:56
@summary: For text to tag. Use BMES tag, S:0, B:1, M:2, E:3
@author: egg
'''

fin = file("../data/pku_valid_ws.ltxt")
text = unicode(fin.read(), "utf-8")
fin.close()

tout = ""

lines = text.split("\n")
lout = []

for line in lines:
	# split to tokens
	tokens = line.split('  ')
	# for out put tags
	tagline = []
	for token in tokens:
		if len(token) == 1:
			tagline.append("0")
		elif len(token) > 1:
			tagline.append("1")
			slen = len(token) - 2
			tagline.extend(["2" for x in xrange(slen)])
			tagline.append("3")
		# tagline.append("  ")

	lout.append("".join(tagline))

fout = file("../data/pku_vaild_ws_tag.ltxt", "wb")
fout.write("\n".join(lout))
fout.close()

