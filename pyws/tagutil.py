# -*- coding: utf-8 -*-
'''
Created on 2013-11-28 11:56
@summary: For text to tag. Use BMES tag, S:0, B:1, M:2, E:3
@author: egg
'''
import re

def text2tags(srcFile, descFile):
	'''
	@summary: 将标记文本转化为标签ID输出
	'''
	fin = file(srcFile, 'rb')
	text = unicode(fin.read(), "utf-8")
	fin.close()

	tout = ""

	lines = text.split("\n")
	lout = []

	for line in lines:
		line = line.strip()
		# split to tokens
		tokens = re.split(r'\s+', line)
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

	fout = file(descFile, "wb")
	fout.write("\n".join(lout))
	fout.close()

def formtext(text, tags):
	'''
	@summary: 按标签将文本按分词格式输出
	'''
	lidx = 0
	otext = []
	for i in range(len(tags)):
		if tags[i] == 0 or tags[i] == 3:
			otext.append(text[lidx:i+1])
			lidx = i+1

	return "  ".join(otext)


if __name__ == "__main__":
	text2tags("data/datasets/msr_ws_train.ltxt", "data/datasets/msr_ws_train_tag.ltxt")