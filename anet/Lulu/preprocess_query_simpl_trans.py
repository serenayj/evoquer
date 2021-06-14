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
def process_line(t):
    line = t.split("##")[-1]
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
    
test_words = {}


test_lines = open("/home/lzl5409/vpmt-master/data/anet/query_info/val_query_info.json").readlines()
train_lines = open("/home/lzl5409/vpmt-master/data/anet/query_info/train_query_info.json").readlines()

vocabs = [] 
for l in test_lines:
    _id = test_lines.index(l)
    words = process_pip(l)
    test_words[_id] = words
    vocabs.append(words)
    
train_words = {} 
for l in train_lines:
    _id = train_lines.index(l)
    words = process_pip(l)
    train_words[_id] = words
    vocabs.append(words)

vocabs.extend(list(test_words.values()))

vocabs_all = [j for i in vocabs for j in i]
vocabs = set(vocabs_all)

vocab_idx = {k:list(vocabs).index(k)+1 for k in vocabs}

vocab_idx['PAD'] = 0 
vocab_idx['<sos>'] = len(vocab_idx) 
vocab_idx['<eos>'] = len(vocab_idx) 

idx_vocab = {v:k for k,v in vocab_idx.items()}

def label_index(queries, vocab_idx):
    out_idx = {}
    for k,v in queries.items():
        v.insert(0, '<sos>')
        v.append('<eos>')
        val = [vocab_idx[i] for i in v]
        out_idx[k] = val 
    return out_idx 

train_idx = label_index(train_words, vocab_idx)

test_idx = label_index(test_words, vocab_idx)

import json 
with open('train_translate.json', 'w') as f:
    json.dump([train_idx, train_words], f)

with open('test_translate.json', 'w') as f:
    json.dump([test_idx, test_words], f)
    
with open('vocab_translate.json', 'w') as f:
    json.dump([vocab_idx, idx_vocab], f)
    
