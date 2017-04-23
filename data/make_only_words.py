import re

f = open('chap5_text_full', 'r')

sentences = ""

for i in f:
	sentences = sentences + " " + i

f.close()
	
pattern = re.compile(r'\s+')

sentences = re.sub(pattern, ' ', sentences)
sentences = sentences.split(' ')

word_list = list()

for i in sentences:
	_i = i.replace('?', '').replace('!', '').replace(',', '').replace('.', '').replace('\"', '')
	_i = _i.replace("''", '').replace(';','').replace(':','').replace('``','').replace(')','')
	_i = _i.replace('(', '').replace(']', '').replace('[', '')
	_i = _i.lower()
	
	if _i != '' and _i != "-----------------------------------------------------------------------":
		word_list.append(_i)

f = open('original_text_for_word', 'w')
for i in word_list:
	f.write(i + ' ')
	
f.close()