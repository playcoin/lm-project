# -*- coding: utf-8 -*-
'''
Created on 2013-12-23 12:19
@summary: POS相关的标签辅助函数
@author: Playcoin
'''

from fileutil import readFile, writeFile
import re

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

	print tagmap["w_b"]
	# olines = []
	# otags = []
	# for line in lines:
	# 	fi, se = procline(line)
	# 	olines.append(fi)
	# 	otags.append(se)

	# otextfile = "data/datasets/pku_pos_train.ltxt"
	# otagfile = "data/datasets/pku_pos_train_tag.ltxt"

	# writeFile(otextfile, '\n'.join(olines))
	# writeFile(otagfile, '\n'.join(otags))


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


if __name__ == "__main__":
	main()