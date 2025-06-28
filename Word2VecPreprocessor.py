import gensim.downloader
import gensim
import numpy as np

model = gensim.downloader.load('fasttext-wiki-news-subwords-300')

def preprocess(word):
	return model[word] if word in model else np.zeros(1000)