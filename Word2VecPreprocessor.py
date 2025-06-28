import gensim.downloader
import gensim
import numpy as np

# fasttext-wiki-news-subwords-300 (big)
# glove-wiki-gigaword-100 (small)
model = gensim.downloader.load('glove-wiki-gigaword-100')

def preprocess(word):
	return model[word] if word in model else np.zeros(len(model['hi']))