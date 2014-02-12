# -*- coding: utf-8 -*-
'''
Created on 2014-01-21 12:19
@summary: NER相关的标签辅助函数
@author: Playcoin
'''
from fileutil import readFile, writeFile
import re

# 标签列表
taglist = ["nr_b","nr_i","nr_e","nt_b","nt_i","nt_e","ns_b","ns_i","ns_e","nz_b","nz_i","nz_e","o"]

tagsize = len(taglist)

tagmap = {}
count = 0
for tag in taglist:
	tagmap[tag.lower()] = count
	count += 1
print "Tag size is:", len(taglist)

############
# Main opr #
############
def main():
	gold_text = readFile("data/datasets/pku_pos_gold_s.ltxt")
	lines = gold_text.split('\n')#[:10]

	olines = []
	otags = []
	for line in lines:
		fi, se = procline(line)
		olines.append(fi)
		otags.append(se)

	# print '\n'.join(olines)
	# print '\n'.join(otags)
	otextfile = "data/datasets/pku_ner_train.ltxt"
	otagfile = "data/datasets/pku_ner_train_tag.ltxt"

	writeFile(otextfile, '\n'.join(olines))
	writeFile(otagfile, '\n'.join(otags))

def nrreplfunc(matchobj):
	return '[' + matchobj.group(0).replace('/nr', '').replace(' ', '') + ']nr'

def ntreplfunc(matchobj):
	return '[' + matchobj.group(1) + ']nt'

def nsreplfunc(matchobj):
	return '[' + matchobj.group(1) + ']ns'

def nzreplfunc(matchobj):
	return '[' + matchobj.group(1) + ']nz'

def ltreplfunc(matchobj):
	# return '[' + re.sub(r'[a-zA-z/ ]', '', matchobj.group(0)) + ']'
	return '[' + re.sub(r'(/[a-zA-z]+)| ', '', matchobj.group(1)) + ']'

def ilreplfunc(matchobj):
	# return '[' + re.sub(r'[a-zA-z/ ]', '', matchobj.group(0)) + ']'
	return matchobj.group(1)

def combinetoken(text):
	'''
	@summary: combine continuous 'nr' token and tokens in '[,]' pair
	'''
	text = re.sub(r'\[([^\]]+)\](?:i|l)', ilreplfunc, text)

	text = re.sub(r'\[([^\]]+)\]', ltreplfunc, text)

	text = re.sub(r'\S+/nr(?:\s+\S+/nr)*', nrreplfunc, text)
	text = re.sub(r'(\S+)/ns', nsreplfunc, text)
	text = re.sub(r'(\S+)/nt', ntreplfunc, text)
	text = re.sub(r'(\S+)/nz', nzreplfunc, text)

	return text

def procline(text):
	'''
	@summary: 将标注数据转为文本串和标签串
	'''
	# print text
	text = combinetoken(text)
	# print text
	tokens = re.split(r"\s+", text)

	ostr = []	# 输出的文本串
	otag = []	# 输出的标签串

	out = str(tagmap['o'])

	for token in tokens:
		if token == "":
			continue
		try:
			# 用反斜杠分开
			fi, se = re.split(r"/|\]", token)
		except:
			print token
			print text

		if '[' not in fi:
			tags = [out for x in range(len(fi))]
		else:
			fi = fi[1:]
			se = se.lower()
			se_b = str(tagmap[se + "_b"])
			se_i = str(tagmap[se + "_i"])
			se_e = str(tagmap[se + "_e"])
			tags = [se_b]
			if len(fi) > 1:
				slen = len(fi) - 2
				tags.extend([se_i for x in range(slen)])
				
				tags.append(se_e)

		assert len(fi) == len(tags)
		ostr.append(fi)
		otag.append(' '.join(tags))

	return ''.join(ostr), ' '.join(otag)


if __name__ == "__main__":
	main()