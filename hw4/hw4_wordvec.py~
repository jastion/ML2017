#coding: utf-8
import os
import sys
import word2vec
import nltk
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
from adjustText import adjust_text

np.set_printoptions(suppress=True)

#set word2vec variables
sizeVectors = 50
downSamplingHFWords = '1e-5'
iterations = 300
min_count = 500

dataRatio = 1.0

def generate_word_data(path):
	directory = os.path.dirname(os.path.abspath("__file__"))
	
	textNames = [path+"/Book 1 - The Philosopher's Stone_djvu.txt",\
				path+"/Book 2 - The Chamber of Secrets_djvu.txt",\
				path+"/Book 3 - The Prisoner of Azkaban_djvu.txt",\
				path+"/Book 4 - The Goblet of Fire_djvu.txt",\
				path+"/Book 5 - The Order of the Phoenix_djvu.txt",\
				path+"/Book 6 - The Half Blood Prince_djvu.txt",\
				path+"/Book 7 - The Deathly Hallows_djvu.txt"]

	with open(directory+"/hw4_word_data.txt",'w') as output:
		for fname in textNames:
			with open(fname) as input:
				output.write(input.read())

	return (directory+"/hw4_word_data.txt")

if __name__ == "__main__":
	vocabulary = []
	vec = []

	data_path = generate_word_data(sys.argv[1])
	print(data_path)

	word2vec.word2vec(data_path, sys.argv[1]+"/word_data.bin",size = sizeVectors,\
						sample = downSamplingHFWords,iter_= iterations,min_count = min_count,verbose = True)
	print("")
	model = word2vec.load(sys.argv[1]+"/word_data.bin")

	print("Orig Size: " + str(model.vectors.shape))

	for word in model.vocab:
		vocabulary.append(word)
		vec.append(model[word])
	vecLength = len(vec)
	vec = np.array(vec)[:int((dataRatio*vecLength))]
	vocabulary = vocabulary[:int((dataRatio*vecLength))]

	tsne = TSNE(n_components=2, random_state=0)
	yTransform = tsne.fit_transform(vec)
	print("New Size: " + str(model.vectors.shape))

	tags = set(['JJ', 'NNP', 'NN', 'NNS'])
	puncts = ["'", ".", ":", ";", ",", "?", "!", u"’"]
	plt.figure()
	texts = []
	nltk.download(['averaged_perceptron_tagger', 'maxent_treebank_pos_tagger', 'punkt'])

	for i, label in enumerate(vocabulary):
		pos = nltk.pos_tag([label])
		if (label[0].isupper() and len(label) > 1 and pos[0][1] in tags
				and all(c not in label for c in puncts)):
			x, y = yTransform[i, :]
			texts.append(plt.text(x, y, label))
			plt.scatter(x, y)

	adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=1.5))

	plt.savefig('hp_result.png', dpi=600)
	#plt.show()
