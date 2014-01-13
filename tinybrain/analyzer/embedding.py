from matplotlib.mlab import PCA
import numpy
import cPickle
import json
import matplotlib.pyplot as plt

import tinybrain.model.parser as parser
from fileutil import readFile, readClearFile
from nlpdict import NlpDict

def load_embedding(emb_file_path):
	f = open(emb_file_path)
	embvalues = cPickle.load(f)
	f.close()
	return embvalues


def pca_word_embedding(embeddings):
	# arr = numpy.array(model['wordEmbedding']['matrix']['data'])
	# arr = arr.transpose()
	return PCA(embeddings)


def rank_word(text, total_word_counts, pos):
	counts = parser.parse_data(text, pos)
	# print counts
	adjusted_counts = {}
	for word in counts.keys():
		# exclude words that are too rare
		if total_word_counts[word] < 50:
			continue
		adjusted_counts[word] = float(counts[word]) / total_word_counts[word]
	sorted_counts = sorted(adjusted_counts.items(), key=lambda x: x[1], reverse=True)
	common_words = {}
	for i in range(len(sorted_counts)):
		common_words[sorted_counts[i][0]] = i+1
	# print(common_words)
	return common_words


def plot_common_words(pca, nlpdict, text, all_pos, colors=['r', 'g', 'b', 'c', 'm', 'y', 'k']):
	# get word ranks for all pos category
	total_word_counts = parser.parse_data(text)
	word_ranks = []
	for pos in all_pos:
		ranks = rank_word(text, total_word_counts, pos)
		word_ranks.append(ranks)

	# assign word to different pos categories
	selected_words = [[] for i in range((len(all_pos) + 1))]

	for word in nlpdict.ndict:
		min_rank = len(nlpdict)
		pos_index = -1
		for i in range(len(word_ranks)):
			rank = 10000000
			if word in word_ranks[i]:
				rank = word_ranks[i][word]
			# omit word that rank too low in a pos category
			if rank > 1000:
				continue
			if rank < min_rank:
				min_rank = rank
				pos_index = i

		if pos_index >= 0:
			selected_words[pos_index].append(word)
		else:
			selected_words[-1].append(word)

	# plot the chart
	selected_words = selected_words[:-1]
	color_index = 0
	for words in selected_words:
		x, y = [], []
		for word in words:
			row = pca.Y[nlpdict[word]]
			x.append(row[0])
			y.append(row[1])
		color = colors[color_index]
		# if words == selected_words[-1]:
			# color = (0.5, 0.5, 0.5, 0.5)
		plt.plot(x, y, '.', color=color)
		color_index += 1

	plt.show()

if __name__ == '__main__':

	nlpdict_text = readClearFile("./data/datasets/pku_ws_train_large.ltxt")
	nlpdict = NlpDict(comb=True, combzh=True, text=nlpdict_text)

	model = load_embedding('data/RnnWFWS2.n_hidden1400.embsize200.in_size4598.embeddings.obj')
	pca = pca_word_embedding(model)

	text = readFile("data/datasets/peoplenew_train.ltxt")
	#plot_common_words(pca, dict, total_word_counts, ['v', 'r'])
	plot_common_words(pca, nlpdict, text, ['a', 'v'], colors=['b', 'm'])
	# plot_common_words(pca, nlpdict, text, ['v', 'r', 'd', 'p', 'a', 'u'])
	#plot_common_words(pca, dict, total_word_counts, ['n', 'v'])
	#plot_common_words(pca, dict, total_word_counts, ['v', 'a'])
	#plot_common_words(pca, dict, total_word_counts, ['v', 'r', 'd', 'p', 'a', 'u'])
	#plot_common_words(pca, dict, total_word_counts, ['n', 'v'])
