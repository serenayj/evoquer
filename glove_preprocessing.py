glove = open("/Users/yanjungao/Desktop/Research/NLP/rl-dep-edu/glove.6B.300d.txt").readlines()

glove_words = {}
for item in glove:
	k = item.split()[0]
	v = list(map(float,item.split()[1:]))
	glove_words[k]=v

glove_db = {} 
sid_gl = [1.0] + [0.0]*299
eid_gl = [0.0]*299 + [1.0]
unk_ = [0.0] + [1.0] + [0.0]*298
pad = [0.0]*300
for k in train_D.wtoi:
	kk = train_D.wtoi[k]
	#print(" idx ",kk)
	if kk == 0: # pad 
		glove_db[kk] = pad 
	elif kk == 2:
		glove_db[kk] = sid_gl 
	elif kk == 3:
		 glove_db[kk] = eid_gl
	elif k in glove_words.keys():
		print("FOUND MATCH  ", k)
		if not list(glove_words[k]):
			glove_db[kk] = [0.0] + [1.0] + [0.0]*298 
		else:
			glove_db[kk] = list(glove_words[k])
	else:
		glove_db[kk] = [0.0] + [1.0] + [0.0]*298 
glove_wtoi = train_D.wtoi 
dics = {"itoglove": glove_db, "wtoi":glove_wtoi}
import json 
with open("glove_db.json", "w") as outf:
	json.dump(dics, outf)