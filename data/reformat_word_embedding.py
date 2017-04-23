import re

f = open('word_embedding')

contents = ""
for i in f:
	contents = contents + i

f.close()
	
	
contents = contents.replace('\n' , '')
contents = contents.split(']')

embeddings = list()
pattern = re.compile(r'\s+')
	
for i in contents:
	i = i.strip()

	i = i.replace('[', '').strip()
	i = re.sub(pattern, ' ', i)
	embeddings.append(i)
	

	
f = open('new_word_embedding', 'w')

for i in embeddings:
	f.write(i)
	f.write('\n')
	
f.close()