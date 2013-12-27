from nlpdict import NlpDict
# from pyws import RnnWS
# from pyws import RnnWFWS, RnnWFWS2, RnnWBWF2WS, RnnRevWS2
# from pylm import RnnEmbTrLM
from fileutil import readClearFile, writeFile

import numpy
import time
import re
import random
# import theano.sandbox.cuda
import cPickle


#############
# Datafiles #
#############
# PKU small valid set
# train_text = readClearFile("./data/datasets/pku_ws_train_large.ltxt")
# nlpdict = NlpDict(comb=True, combzh=True, text=train_text)

test_text = readClearFile("./data/datasets/pku_ws_test.ltxt")
test_tags = readClearFile("./data/datasets/pku_ws_test_tag.ltxt")

# ws = RnnWFWS2(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5,# emb_dr_rate=0.1,
# 	backup_file_path="./data/model/RnnWFWS2.model.epoch60.n_hidden1400.ssl20.truncstep4.dr0.5.embsize200.in_size4598.rc91.obj"
# )

# wsrev = RnnRevWS2(nlpdict, n_emb=200, n_hidden=1400, lr=0.5, batch_size=150, 
# 	l2_reg=0.000001, truncate_step=4, train_emb=True, dr_rate=0.5,# emb_dr_rate=0.1,
# 	backup_file_path="./data/model/RnnRevWS2.model.epoch56.n_hidden1400.ssl20.truncstep4.dr0.5.embsize200.in_size4598.rc91.obj"
# )


f = file("./data/test_records.obj", 'rb')
records = cPickle.load(f)
f.close()


def main():
	stime = time.clock()
	sents = test_text.split('\n')
	taglines = test_tags.split('\n')

	ct = 0
	cw = 0
	cfr = 0
	cfw = 0
	crr = 0
	crw = 0
	# samples = random.sample(range(len(sents)), 500)
	for i in range(len(sents)):
		text = sents[i]
		tags = [int(x) for x in taglines[i]]
		# tags1, pm1 = ws.segdecode(text, decode=True)
		# tags2, pm2 = wsrev.segdecode(text, decode=True, rev=True)

		# gps = findDiff(tags1, tags2)

		# records.append([])
		tags1, pm1, tags2, pm2, gps = records[i]
		text_len = len(text)
		for j in range(len(gps)):
			pair = gps[j]
			a = tags1[pair[0]:pair[1]+1]
			b = tags2[pair[0]:pair[1]+1]
			c = tags[pair[0]:pair[1]+1]
			s1, s2 = calDiff(pair, tags1, pm1, tags2, pm2)
			d1, d2 = calDiff2(j, gps, text_len)
			s1 += 0.005 * d1
			s2 += 0.005 * d2
			d = s1 >= s2 and 'f' or 'r'

			if a == c:
				if s1 >= s2:
					cfr += 1
				else:
					crw += 1
			elif b == c:
				if s1 >= s2:
					cfw += 1
				else:
					crr += 1
			else:
				cw += 1

			print text[pair[0]:pair[1]+1], a, b, c, d, "%.5f %.5f %.5f %.5f" % (s1, s2, d1, d2)
			ct += 1

	# f = file("./data/test_records.obj", 'wb')
	# cPickle.dump(records, f)
	# f.close()
	print "Total %d, TW %d, FR %d, FW %d, RR %d, RW %d" % (ct, cw, cfr, cfw, crr, crw)
	print "Total time: %.1fm" % ((time.clock() - stime) / 60.)

# find the different
def findDiff(tags1, tags2):

	last = -1
	groups = []
	pair = []
	for i in range(len(tags1)):
		if tags1[i] != tags2[i]:
			if i != last+1:
				if len(pair) == 1:
					pair.append(last)
					groups.append(pair)
				pair = [i]
			last = i

	if len(pair) == 1:
		pair.append(last)
		groups.append(pair)

	return groups

def calDiff(pair, tags1, pm1, tags2, pm2):
	"find the surround probs"
	# sum1 = 0.
	# sum2 = 0.
	sidx = min(findPre(tags1, pair[0]), findPre(tags2, pair[0]))
	eidx = max(findSuf(tags1, pair[1]), findSuf(tags2, pair[1]))
	# for i in range(sidx, eidx):
		# sum1 += pm1[i][tags1[i]]
		# sum2 += pm2[i][tags2[i]]

	tsum1 = 0.
	tsum2 = 0.
	# sidx = max(findPre(tags1, sidx-1), findPre(tags2, sidx-1))
	# sidx = max(findPre(tags1, sidx-1), findPre(tags2, sidx-1))
	sidx = max(sidx - 2, 0)
	eidx = min(eidx + 2, len(tags1))
	for i in range(sidx, eidx):
		tsum1 += pm1[i][tags1[i]]
		tsum2 += pm2[i][tags2[i]]

	return tsum1, tsum2

def calDiff2(idx, groups, text_len):
	"find the nearest different probs"

	# d1 = idx > 0 and groups[idx-1][1] or 0
	# d2 = (idx < len(groups) - 1) and groups[idx+1][0] or text_len
	# d1 = groups[idx][0] - d1
	# d2 = d2 - groups[idx][1] 

	d1 = max(groups[idx][0] - 0, 80)
	d2 = max(text_len - groups[idx][1], 80)
	s = float(d1 + d2)
	d1 = numpy.log(d1 / s)
	d2 = numpy.log(d2 / s)

	return d1, d2

def findPre(tags, idx):
	if idx < 0:
		return 0

	if tags[idx] == 2 or tags[idx] == 3:
		while idx >= 0 and tags[idx] != 1:
			idx -= 1

	return max(idx, 0)

def findSuf(tags, idx):
	# idx + 1
	len_t = len(tags)
	if idx >= len_t:
		return len_t

	if tags[idx] == 1 or tags[idx] == 2:
		while idx < len_t and tags[idx] != 3:
			idx += 1

	return min(idx+1, len_t)


if __name__ == "__main__":
	main()