import tensorflow as tf
import numpy as np
#read embedding


def get_wordList():
	f = open('./data/word', 'r')
	wordList = list()
	
	for l in f:
		wordList.append(l.strip())
	
	return wordList

	
def get_wordEmbeddings():
	embedding = np.loadtxt('./data/word_embedding')
	#embedding = np.loadtxt('./data/data_word_test.txt')
	return embedding


def get_embeddingLookup():
	wordList = get_wordList()
	wordEmbeddings = get_wordEmbeddings()
	
	word2id = {w:i for i, w in enumerate(wordList)}	
	word2em = {word:embedding for word, embedding in zip(wordList, wordEmbeddings)}
	return word2em, word2id

