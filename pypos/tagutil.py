# -*- coding: utf-8 -*-
'''
Created on 2013-12-23 12:19
@summary: POS相关的标签辅助函数
@author: Playcoin
'''

from fileutil import readFile, writeFile
import re
import os

# 标签列表
taglist = ["Ag_b","Ag_i","Ag_e","a_b","a_i","a_e","ad_b","ad_i","ad_e","an_b","an_i","an_e",\
	"b_b","b_i","b_e","bg_b","bg_i","bg_e","c_b","c_i","c_e","dg_b","dg_i","dg_e","d_b","d_i","d_e",\
	"e_b","e_i","e_e","f_b","f_i","f_e","g_b","g_i","g_e","h_b","h_i","h_e",\
	"i_b","i_i","i_e","j_b","j_i","j_e","k_b","k_i","k_e","l_b","l_i","l_e",\
	"m_b","m_i","m_e","mg_b","mg_i","mg_e","Ng_b","Ng_i","Ng_e","n_b","n_i","n_e","nr_b","nr_i","nr_e",\
	"ns_b","ns_i","ns_e","nt_b","nt_i","nt_e","nz_b","nz_i","nz_e","nx_b","nx_i","nx_e","o_b","o_i","o_e",\
	"p_b","p_i","p_e","q_b","q_i","q_e","r_b","r_i","r_e","rg_b","rg_i","rg_e","s_b","s_i","s_e",\
	"tg_b","tg_i","tg_e","t_b","t_i","t_e","u_b","u_i","u_e","vg_b","vg_i","vg_e",\
	"v_b","v_i","v_e","vd_b","vd_i","vd_e","vn_b","vn_i","vn_e","w_b","w_i","w_e",\
	"x_b","x_i","x_e","y_b","y_i","y_e","z_b","z_i","z_e","un_b","un_i","un_e"\
]

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
	lines = gold_text.split('\n')

	olines = []
	# otags = []
	for line in lines:
	# 	fi, se = procline(line)
	# 	olines.append(fi)
	# 	otags.append(se)
		olines.append(clearner(line))

	# otextfile = "data/datasets/pku_pos_train.ltxt"
	# otagfile = "data/datasets/pku_pos_train_tag.ltxt"

	# writeFile(otextfile, '\n'.join(olines))
	# writeFile(otagfile, '\n'.join(otags))
	writeFile("data/datasets/pku_pos_gold_train.ltxt", '\n'.join(olines))

def clearner(text):
	tokens = re.split(r"\s+", text)

	ostr = []	# 输出的文本串
	for token in tokens:
		if token == "":
			continue

		# 用反斜杠分开
		fi, se = token.split("/")
		fi = fi[0] == '[' and fi[1:] or fi
		se = "]" in se and se.split(']')[0] or se

		ostr.append("%s/%s" % (fi, se))

	return '  '.join(ostr)

def procline(text):
	'''
	@summary: 将标注数据转为文本串和标签串
	'''

	tokens = re.split(r"\s+", text)

	ostr = []	# 输出的文本串
	otag = []	# 输出的标签串
	for token in tokens:
		if token == "":
			continue

		# 用反斜杠分开
		fi, se = token.split("/")
		fi = fi[0] == '[' and fi[1:] or fi
		se = "]" in se and se.split(']')[0] or se

		se_b = str(tagmap[(se + "_b").lower()])
		se_i = str(tagmap[(se + "_i").lower()])
		se_e = str(tagmap[(se + "_e").lower()])


		tags = [se_b]
		if len(fi) > 1:
			slen = len(fi) - 2
			tags.extend([se_i for x in range(slen)])
			
			tags.append(se_e)

		ostr.append(fi)
		otag.append(' '.join(tags))

	return ''.join(ostr), ' '.join(otag)

def formtext(text, tags):
	'''
	@summary: 按标签将文本按分词格式输出
	'''
	tokens = []
	wtags = []

	token = text[0]
	wtag = taglist[tags[0]].lower()

	for i in range(1, len(text)):
		if tags[i] % 3 == 0:
			tokens.append(token)
			wtags.append(wtag.split('_')[0])
			token = ""
			wtag = taglist[tags[i]].lower()
		token = token + text[i]

	if token != "":
		tokens.append(token)
		wtags.append(wtag.split('_')[0])

	otokens = []
	for (token, wtag) in zip(tokens, wtags):
		otokens.append("%s/%s" % (token, wtag))

	return "  ".join(otokens)


def score(train_text, text1, text2):

	d = set(re.split(r'\s+', train_text))

	lines1 = text1.strip().split('\n')
	lines2 = text2.strip().split('\n')

	assert len(lines1) == len(lines2)
	err = 0
	iv = 0
	oov = 0
	oovmiss = 0
	ivmiss = 0
	truewords = 0
	testwords = 0
	good = 0

	for line1, line2 in zip(lines1, lines2):
		words1 = re.split(r'\s+', line1)
		words2 = re.split(r'\s+', line2)
		truewords += len(words1)
		testwords += len(words2)
		ot1 = '\n'.join(words1)
		ot2 = '\n'.join(words2)
		writeFile('pypos/tmp1', ot1)
		writeFile('pypos/tmp2', ot2)
		os.system("diff -y -i pypos/tmp1 pypos/tmp2 > pypos/tmpo")
		lines = readFile("pypos/tmpo").strip().split('\n')
		for line in lines:
			# total
			if re.search(r'\s[\|\>\<]\s', line):
				err += 1
			else:
				good += 1

			# check oov
			m = re.search(r'^([^\s]+)\s', line) 
			if m:
				word = m.group(1)
				if word in d:
					iv += 1
				else:
					oov += 1
				# not 'insert' line
				if re.search(r'^[^\s]+\s+[\|\>\<]\s', line):
					if word in d:
						ivmiss += 1
					else:
						oovmiss += 1

	os.remove("pypos/tmp1")
	os.remove("pypos/tmp2")
	os.remove("pypos/tmpo")
	p = float(good) / testwords
	r = float(good) / truewords
	print "OOV Rate:", float(oov) / truewords
	if oov > 0:
		print "OOV Recall:", 1 - float(oovmiss) / oov
	print "IV Recall:", 1 - float(ivmiss) / iv
	print "Precision:", p
	print "Recall:", r
	print "Total F1:", (2*p*r) / (p+r) 

if __name__ == "__main__":
	# main()
	t = readFile('data/datasets/pku_pos_gold_train.ltxt')
	t1 = readFile('data/datasets/pku_pos_gold_test.ltxt')
	t2 = readFile('pypos/o1400_rev.ltxt')
	score(t, t1, t2)
