from __future__ import print_function
import tensorflow as tf
import word_embedding as wem
import sentence_encoding as sn
import time
import os
import sys
import numpy as np
from model import Model
import datetime as dt


def train():
	print("start training!!")
	
	restore = True
	epochs = 1000
	batch_size = 1

	args = sys.argv
	args = args[1:]
	
	for _i in range(int(len(args)/2)):
		arg_idx = _i * 2
		val_idx = _i * 2 + 1
		
		arg, value = args[arg_idx], args[val_idx]
		
		if arg == '-r':
			restore = value
		elif arg == '-e':
			epochs = int(value)
		elif arg == '-b':
			batch_size = int(value)
		

	print(restore, epochs, batch_size)
	

	
	params = dict()
	seq_length = 25
	
	
	rnn_size = 6
	params['rnn_size'] = rnn_size
	params['seq_length'] = seq_length
	#get word embedding
	word2em, word2id = wem.get_embeddingLookup()
	total_word_cnt = len(word2em)
	print("--------------------------")
	print(len(word2em))
	print(len(word2id))
	print(total_word_cnt)
	print("--------------------------")
	
	params['total_word_cnt'] = total_word_cnt
	params['total_size'] = len(sn.get_sentences())
	
	#encoding setneces
	em_encoded_sentences = sn.get_encoded_sentences(word2em, seq_length)
	id_encoded_sentences = sn.get_encoded_sentences(word2id, seq_length)
	em_encoded_sentences_for_decoder = sn.get_encoded_sentences_for_decoder(word2em, seq_length)

	print("--------------------------")
	print("--------------------------")
	print("--------------------------")
	
	print(em_encoded_sentences[0])
	print(id_encoded_sentences[0])
	print(len(id_encoded_sentences))
	
	print("--------------------------")
	print("--------------------------")
	print("--------------------------")
	
	
	total_size = len(em_encoded_sentences)
	params['batch_size'] = batch_size
	
	model = Model(params)
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		with tf.device("/gpu:0"):	
			summaries = tf.summary.merge_all()
			#writer = tf.summary.FileWriter('./graph', time.strftime("%Y-%m-%d-%H-%M-%S"))
			#writer.add_graph(sess.graph)
			
			sess.run(tf.global_variables_initializer())
			
			model_ckpt_file = './status/model.ckpt'
			#if os.path.isfile(model_ckpt_file):
			if restore == True:
				saver.restore(sess, model_ckpt_file)
			
			for e in range(epochs):
				# get bunch of data
				start_time = dt.datetime.now()
				
				start = 0
				for i in range(int(total_size/batch_size) + 1):
					if i*batch_size >= 1200:
						break
					start_time_in = dt.datetime_now()
					
					print("processing", i)
					end = start + batch_size
					
					if i == total_size/batch_size:
						if start > total_size - 1:
							break
						else:
							end = total_size - 1
						
						
					#1. encoder_final_states
					em_en_tran = np.transpose(em_encoded_sentences[start:end], (1,0,2))
					em_de_tran = np.transpose(em_encoded_sentences_for_decoder[start:end], (1,0,2))
					id_en_tran = np.transpose(id_encoded_sentences[start:end])
					
					feed = {model.encoder_input:em_encoded_sentences[start:end]}
					
					encoder_states = sess.run(model.encoder_final_states, feed_dict=feed)
					
					#feed_state = dict()
					#feed_state[model.decorder_initial_state] = encoder_states
					#******************************************************************************
					for r in range(end-start):
						#model.C[start + r] = encoder_states[r]
						sess.run(model.C[start + r].assign(encoder_states[r]))
					#******************************************************************************
						
					feed_decoder_input = dict()
					feed_decoder_target = dict()
					
					
					for i in range(seq_length):
						feed_decoder_input[model.decoder_input[i]] = em_de_tran[i]
						feed_decoder_target[model.decoder_target[i]] = id_en_tran[i]
					
					feed_decoder = dict()
					feed_decoder = feed_decoder_input
					feed_decoder.update(feed_decoder_target)
					#feed_decoder.update(feed_state)
					feed_decoder.update(feed)
					_cost, _ = sess.run([model.cost, model.train], feed_dict=feed_decoder)

					start = end
					print("Take", str((dt.datetime.now() - start_time_in).seconds), "seconds for ", batch_size, "batch_size")
					
				print("Take", str((dt.datetime.now() - start_time).seconds), "seconds for 1 cycles")
				
			
				saver.save(sess, model_ckpt_file)
				print("mode saved to ", model_ckpt_file)
				print("%d, %d" % (i, e))
				print("the cost = ", _cost)
			
			print(model.C)
			print(sess.run(model.C))		

		
if __name__ == '__main__':
	train()
