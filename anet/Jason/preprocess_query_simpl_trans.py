import benepar 
parser = benepar.Parser("benepar_en3")
def parse_line(line):
    tree = parser.parse(line)
    item = None
    for tr in tree.subtrees():
        if tr.label() == 'VP':
            item = tr
    return item 

# Process queries by constituency parser, extract main verb (verb not in subordinating conjunction)
# Main objects are also extracted 
t = "person turns off the light as they're leaving"

from nltk.tree import ParentedTree
def is_sbar(tr, v_cnt):
    ptr = tr.parent()
    flag = False 
    while ptr:
        if ptr.label() == 'SBAR' and v_cnt !=0:
            flag = True 
            break 
        else:
            ptr = ptr.parent() 
    return flag 

verbs_tags = ['VBZ', 'VBP', 'VB', 'VBD','VBG', 'VBN']
nouns_tags = ['NNS', 'NN']
def find_verb(item):
    verb = None 
    for st in item.subtrees():
        if st.label() in verbs_tags:
            verb = st.leaves()
            return verb, st  
    return verb, st  

def find_noun(item):
    noun = None 
    for st in item.subtrees():
        if st.label() in nouns_tags:
            noun = st.leaves()
            return noun
    return None 

# Extract main verbs and nouns as translation phrase 
def process_line(line):
    #line = t.split("##")[-1]
    tree = parser.parse(line)
    newtree = ParentedTree.convert(tree)
    output= [] 
    verbs = [] 
    nouns = [] 
    v_cnt = 0 
    out = ""
    for tr in newtree.subtrees():
        if tr.label() == 'VP':
            #flag = is_sbar(tr, v_cnt)
            flag = False 
            if not flag:
                verb, st = find_verb(tr)
                if verb not in verbs:
                    verbs.append(verb)
                    v_cnt +=1 
                    if verb:
                        out += " "+ " ".join(verb)
                noun = find_noun(tr)
                if noun not in nouns: 
                    nouns.append(noun)
                    #print(noun)
                    if noun:
                        out += " "+ " ".join(noun)
    return verbs, nouns, out   

verbs, nouns, out  = process_line(t)
stem_queries_verb = {}

from nltk.stem.wordnet import WordNetLemmatizer

out.split()
words = [WordNetLemmatizer().lemmatize(w,'v') for w in out.split()]

def process_pip(line):
    verbs, nouns, out  = process_line(line)
    words = [WordNetLemmatizer().lemmatize(w,'v') for w in out.split()]
    return words 

vocabs = []
test_words = {}
train_words = {}

#Build vocabulary from both test and train:
import json
with open("/home/lzl5409/vpmt-master/data/anet/query_info/val_query_info.json") as jsonfile:
    test_lines = json.load(jsonfile)
with open("/home/lzl5409/vpmt-master/data/anet/query_info/train_query_info.json") as jsonfile:
    train_lines= json.load(jsonfile)

wtoi1 = test_lines['wtoi']
vocabs = list(wtoi1.keys())

wtoi2 = train_lines['wtoi']
for i in wtoi2:
    vocabs.append(i)

#vocabs.extend(list(test_words.values()))

#vocabs_all = [j for i in vocabs for j in i]
vocabs = set(vocabs)
print(len(vocabs))


vocabs = []
numOfTokens = 0
numOfTokensSimpl = 0
#Build test_words and train_words
train_lines = open("/home/jjw6188/vpmt/data/anet/train_simple_query.txt","r").readlines()
test_lines = open("/home/jjw6188/vpmt/data/anet/val_simple_query.txt","r").readlines()

for line in train_lines:
    qrys = line.split("##")
    qid = qrys[0]
    query = qrys[1].strip()
    words = process_pip(query)
    #print(query)
    #print(words)
    numOfTokens += len(query.split())
    numOfTokensSimpl += len(words)
    #print("numOfTokens: ", numOfTokens)
    #print("numOfTokensSimpl: ", numOfTokensSimpl)
    train_words[qid] = words
    vocabs.append(words)

#print("Train vocab length: ", len(vocabs))

for line in test_lines:
    qrys = line.split("##")
    qid = qrys[0]
    query = qrys[1].strip()
    words = process_pip(query)
    numOfTokens += len(query.split())
    numOfTokensSimpl += len(words)
    test_words[qid] = words
    vocabs.append(words)

numQueries = len(train_lines) + len(test_lines)
print("Average #Tokens per Query (Original):", numOfTokens/numQueries)
print("Average #Tokens per Query (Simplified):", numOfTokensSimpl/numQueries)
'''
i
qid = 0
qids = []
sentences = []
for vid in data.keys():
    ann=data[vid]
    descrip = " ".join(ann['sentences'])
    for sentence in ann['sentences']:
        qids.append(qid)
        sentences.append(sentence)
        qid += 1
print(sentences)
#for linprint("Average #Tokens per Query (Original):", numOfTokens/numQueriese in sentences:
#    " ".join(words)
#print(sentences)
for sentence in ann['sentences']:
    # MAKE QID:
    _id = qids[ann['sentences'].index(sentence)]
    words = process_pip(sentence)
    test_words[_id] = words
'''

vocabs.extend(list(test_words.values()))

vocabs_all = [j for i in vocabs for j in i]
from collections import Counter
vocab_cnt = Counter(vocabs_all)
vocab_match = {x:count for x, count in vocab_cnt.items() if count >= 10}
vocabs = vocab_match.keys()
print("Train+Test vocab length (Simplified)", len(vocabs))

vocab_idx = {k:list(vocabs).index(k)+1 for k in vocabs}

vocab_idx['PAD'] = 0 
vocab_idx['<sos>'] = len(vocab_idx) 
vocab_idx['<eos>'] = len(vocab_idx) 
vocab_idx['<unk>'] = len(vocab_idx)

idx_vocab = {v:k for k,v in vocab_idx.items()}

def label_index(queries, vocab_idx):
    out_idx = {}
    for k,v in queries.items():
        val = []
        v.insert(0, '<sos>')
        v.append('<eos>')
        for i in v:
            if i in vocab_idx:
                val.append(vocab_idx[i])
            else:
                val.append(vocab_idx["<unk>"])
        #val = [vocab_idx[i] for i in v]
        out_idx[k] = val 
    return out_idx 

train_idx = label_index(train_words, vocab_idx)

test_idx = label_index(test_words, vocab_idx)
 
with open('anet_train_translate.json', 'w') as f:
    json.dump([train_idx, train_words], f)

with open('anet_test_translate.json', 'w') as f:
    json.dump([test_idx, test_words], f)
    
with open('anet_vocab_translate.json', 'w') as f:
    json.dump([vocab_idx, idx_vocab], f)   
