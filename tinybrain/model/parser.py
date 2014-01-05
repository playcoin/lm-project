__author__ = 'eric'

from collections import Counter
from fileutil import readFile
import re


def parse_data(text, pos=None):
	pos = pos is not None and pos.lower() or None

	word_counts = Counter()

	lines = text.split('\n')

	for line in lines:
		tokens = re.split(r'\s+', line.strip())[1:]
		for token in tokens:
			fi, se = token.split('/')
			if pos is None or se.lower() == pos:
				for c in fi:
					word_counts.update({c, 1})

	return word_counts

if __name__ == '__main__':
	text = readFile("data/datasets/peoplenew_train.ltxt")

	counts = parse_data(text[:200])
	print(counts)
