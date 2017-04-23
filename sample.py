from __future__ import print_function
import tensorflow as tf
import os
from model import Model
import word_embedding as wem
import sentence_encoding as sn
import numpy as np
import sys


def sample():
	
	test_state_num = 0
	
	args = sys.argv
	args = args[1:]
		
	for _i in range(int(len(args)/2)):
		arg_idx = _i * 2
		val_idx = _i * 2 + 1
		
		arg, value = args[arg_idx], args[val_idx]
		
		if arg == '-n':
			test_state_num = int(value)
	
	

	params = dict()
	seq_length = 25
	
	rnn_size = 6
	params['rnn_size'] = rnn_size	
	params['seq_length'] = seq_length
	word2em, word2id = wem.get_embeddingLookup()
	total_word_cnt = len(word2em)
	params['total_word_cnt'] = total_word_cnt
	batch_size = 1
	params['batch_size'] = batch_size
	total_size = len(sn.get_sentences())
	params['total_size'] = total_size

	model = Model(params, training=False)
	wordList = wem.get_wordList()
	
	print("before session")
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		model_ckpt_file = './status/model.ckpt'
		saver.restore(sess, model_ckpt_file)
		#if os.path.isfile(model_ckpt_file):
		print("Setting done.")
		
		
		#********************************************************
		print(model.C)
		print(sess.run(model.C))
		
		#print(model.C[1])
		#state = list(sess.run(model.C[1])))
		
		feed_predict = dict()
		feed_decoder_input = dict()
				
		seed_input = [[list(word2em["GO"]) for _ in range(seq_length)]]
		seed_input = np.transpose(seed_input, (1,0,2))
		
		for i in range(seq_length):
			feed_decoder_input[model.decoder_input[i]] = seed_input[i]
				
		feed_predict = feed_decoder_input
		
		state = list(sess.run(model.C[test_state_num]))
		print(state)
		feed_predict[model.decorder_initial_state] = [state]
		#feed_predict[model.decorder_initial_state] = [[-0.57806301, -0.97704571, -0.99187565]]
		
		print("Predict result....")
		
		result = sess.run(model.predict, feed_dict=feed_predict)
		result = np.transpose(result)
		result = list(result[0])
		result = [wordList[id] for id in result]
		result = ' '.join(result)
		print(result)

if __name__ == '__main__':
	sample()	